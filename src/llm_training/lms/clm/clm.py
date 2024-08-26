from typing import Any

import torch
from torchmetrics.text import Perplexity

from llm_training.lms.base_lm import BaseLightningModule
from llm_training.lms.protos import CausalLMProto
from llm_training.lms.utils import get_model
from llm_training.metrics import ConsumedSamples, ConsumedTokens
from llm_training.models.base_model.base_model import BaseModel
from llm_training.ops import cross_entropy, shift_labels

from .clm_config import CLMConfig


class CLM(BaseLightningModule):
    config: CLMConfig
    model: CausalLMProto | BaseModel | None

    def __init__(self, config: CLMConfig) -> None:
        super().__init__(config)

        self.model = None
        self.train_perplexity = Perplexity(ignore_index=self.config.ignore_index)
        self.val_perplexity = Perplexity(ignore_index=self.config.ignore_index)
        self.consumed_samples = ConsumedSamples()
        self.consumed_tokens = ConsumedTokens(ignore_index=self.config.ignore_index)
    
    @property
    def has_pre_trained_weights(self) -> bool:
        if self.model is None:
            return False
        return self.model.has_pre_trained_weights

    def get_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        state_dict = self.model.get_pre_trained_weights()
        state_dict = {f'model.{k}': v for k, v in state_dict.items()}
        return state_dict

    def configure_model(self) -> None:
        self.model = get_model(self.config.model)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return cross_entropy(
            logits,
            labels,
            ignore_index=self.config.ignore_index
        )

    def get_noisy_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeds(input_ids)
        noise = torch.zeros_like(inputs_embeds).uniform_(-1, 1)
        input_lengths = torch.sum(attention_mask, 1)
        delta = noise * attention_mask.unsqueeze(2)
        dims = input_lengths * inputs_embeds.size(-1)
        magnitude = self.config.neftune_alpha / torch.sqrt(dims)
        delta = (delta * magnitude.view(-1, 1, 1)).detach()
        inputs_embeds += delta
        inputs_embeds.requires_grad_()
        return inputs_embeds

    def training_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> torch.Tensor:
        labels = shift_labels(batch['labels'], self.config.ignore_index)
    
        if self.config.neftune_alpha is not None:
            logits = self.model(
                attention_mask=batch['attention_mask'],
                position_ids=batch.get('position_ids', None),
                input_embeds=self.get_noisy_embeddings(batch['input_ids'], batch['attention_mask'])
            )

            self.log('NEFTune Alpha', self.config.neftune_alpha)
        else:
            logits = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                position_ids=batch.get('position_ids', None)
            )

        loss = self.compute_loss(logits, labels)

        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train/Step', loss)

        if self.grad_norm is not None:
            self.log('Gradient Norm', self.grad_norm)

        self.train_perplexity(logits, labels)
        self.log('Perplexity/Train/Step', self.train_perplexity)

        self.consumed_samples.update(labels)
        self.consumed_tokens.update(labels)
        self.logger.log_metrics({
            'Consumed Samples': self.consumed_samples.compute(),
            'Consumed Tokens': self.consumed_tokens.compute()
        })
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int, dataloader_idx: int = 0):
        batch_size = batch['input_ids'].size(0)
        labels = shift_labels(batch['labels'], self.config.ignore_index)
        logits = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch.get('position_ids', None)
        )

        loss = self.compute_loss(logits, labels)

        self.val_perplexity.update(logits, labels)
        self.log('Loss/Val', loss, batch_size=batch_size, sync_dist=True)
        self.log('Perplexity/Val', self.val_perplexity)

    def get_model(self) -> BaseModel:
        return self.model
