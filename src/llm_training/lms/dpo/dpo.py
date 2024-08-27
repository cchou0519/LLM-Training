from typing import Any

import torch
import torch.nn.functional as F

from llm_training.lms.base_lm import BaseLightningModule
from llm_training.lms.protos import CausalLMProto
from llm_training.lms.utils import get_model
from llm_training.models.base_model.base_model import BaseModel
from llm_training.ops.cross_entropy_op import shift_labels

from .dpo_config import DPOConfig


class DPO(BaseLightningModule):
    config: DPOConfig
    model: CausalLMProto | BaseModel | None
    ref_model: CausalLMProto | BaseModel | None


    def __init__(self, config: DPOConfig) -> None:
        super().__init__(config)

        self.model = None
        self.ref_model = None
    
    @property
    def has_pre_trained_weights(self) -> bool:
        if self.model is None:
            return False
        return self.model.has_pre_trained_weights

    def get_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        model_state_dict = self.model.get_pre_trained_weights()
        ref_model_state_dict = (
            model_state_dict if self.config.ref_model is None
            else self.ref_model.get_pre_trained_weights()
        )
        state_dict = {}
        state_dict |= {f'model.{k}': v for k, v in model_state_dict.items()}
        state_dict |= {f'ref_model.{k}': v for k, v in ref_model_state_dict.items()}
        return state_dict
    
    def configure_model(self) -> None:
        self.model = get_model(self.config.model)
        self.ref_model = get_model(self.config.ref_model or self.config.model)
        self.ref_model.eval().requires_grad_(False)

    def on_fsdp_wrap_model(self, state_dict: dict[str, torch.Tensor] | None) -> None:
        assert self.model.no_split_modules
        assert self.ref_model.no_split_modules

        self.model = self.fsdp_wrap_model(
            self.model,
            'model',
            state_dict,
            self.model.no_split_modules
        )
        
        self.ref_model = self.fsdp_wrap_model(
            self.ref_model,
            'ref_model',
            state_dict,
            self.ref_model.no_split_modules,
            training=False
        )
    
    def compute_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> torch.Tensor:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        loss = (
            -F.logsigmoid(self.config.beta * logits) * (1 - self.config.label_smoothing)
            - F.logsigmoid(-self.config.beta * logits) * self.config.label_smoothing
        )
        loss = loss.mean()
        return loss

    def compute_metrics(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        suffix: str = ''
    ) -> dict[str, torch.Tensor]:
        chosen_rewards = self.config.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.config.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        chosen_reward = chosen_rewards.mean()
        rejected_reward = rejected_rewards.mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()

        return {
            f'Chosen Reward{suffix}': chosen_reward,
            f'Rejected Reward{suffix}': rejected_reward,
            f'Reward Accuracy{suffix}': reward_accuracy,
            f'Reward Margin{suffix}': reward_margin,
            f'Chosen Log P{suffix}': policy_chosen_logps.mean(),
            f'Rejected Log P{suffix}': policy_rejected_logps.mean()
        }

    def get_logps(
        self,
        model: CausalLMProto,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        labels = shift_labels(labels, self.config.ignore_index)

        loss_mask = labels != self.config.ignore_index
        index = labels.masked_fill(labels == self.config.ignore_index, 0).unsqueeze(2)
        per_token_logps = logits.log_softmax(-1).gather(2, index).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1)

    def forward_batch(self, batch: dict[str, torch.Tensor | Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_chosen_logps = self.get_logps(
            self.model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['chosen_labels']
        )
        policy_rejected_logps = self.get_logps(
            self.model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch['rejected_labels']
        )
        reference_chosen_logps = self.get_logps(
            self.ref_model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['chosen_labels']
        )
        reference_rejected_logps = self.get_logps(
            self.ref_model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch['rejected_labels']
        )
        return (
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )

    def training_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> torch.Tensor:
        (
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        ) = self.forward_batch(batch)

        loss = self.compute_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )

        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train/Step', loss)

        if self.grad_norm is not None:
            self.log('Gradient Norm', self.grad_norm)

        metrics = self.compute_metrics(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            suffix='/Train/Step'
        )
        
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> None:
        batch_size = batch['chosen_input_ids'].size(0)

        (
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        ) = self.forward_batch(batch)

        loss = self.compute_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        self.log('Loss/Val', loss, sync_dist=True, batch_size=batch_size)

        metrics = self.compute_metrics(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            suffix='/Val'
        )
        self.log_dict(metrics, sync_dist=True, batch_size=batch_size)

    def get_model(self) -> BaseModel:
        return self.model
