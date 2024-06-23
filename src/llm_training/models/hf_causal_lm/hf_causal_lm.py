import logging
from typing import Any, ContextManager

import torch
from accelerate import init_empty_weights
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, wrap
from torchmetrics.text import Perplexity
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel

from llm_training.metrics import ConsumedSamples, ConsumedTokens
from llm_training.models.hf_compat_model import HFCompatModel
from llm_training.ops.cross_entropy_op import cross_entropy, shift_labels
from llm_training.overrides import FSDPStrategy
from llm_training.utils.decorators import copy_method_signature

from .hf_causal_lm_config import HFCausalLMConfig
from .patchers import AutoPatcher

try:
    from peft import get_peft_model # type: ignore
except ImportError:
    ...


logger = logging.getLogger(__name__)


class HFCausalLM(HFCompatModel):
    hf_model_class = AutoModelForCausalLM
    
    config: HFCausalLMConfig
    hf_model: PreTrainedModel

    def __init__(self, config: HFCausalLMConfig) -> None:
        super().__init__(config)

        self.train_perplexity = Perplexity(ignore_index=-100)
        self.val_perplexity = Perplexity(ignore_index=-100)
        self.consumed_samples = ConsumedSamples()
        self.consumed_tokens = ConsumedTokens()

    def convert_state_dict_from_hf(self, hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {'hf_model.' + k: v for k, v in hf_state_dict.items()}

    def convert_state_dict_to_hf(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k.removeprefix('hf_model.'): v for k, v in state_dict.items()}

    def configure_model_context(self) -> ContextManager:
        if isinstance(self.strategy, FSDPStrategy):
            return init_empty_weights(include_buffers=False)
        return super().configure_model_context()

    def configure_model(self) -> None:
        assert not isinstance(self.strategy, FSDPStrategy) or self.strategy.tp_size == 1

        self.hf_model = self.construct_hf_model()
        if self.config.patcher_config != False:
            self.hf_model = AutoPatcher.patch(self.hf_model, self.config.patcher_config)

        if self.global_rank == 0:
            logger.info(f'Config:\n{self.hf_config}')
            logger.info(f'Model:\n{self.hf_model}')
        
        if self.config.enable_gradient_checkpointing:
            self.hf_model.gradient_checkpointing_enable({'use_reentrant': False})

    def _on_configure_model_end_fsdp(self) -> None:
        assert isinstance(self.strategy, FSDPStrategy)

        state_dict = None
        if self.global_rank == 0 and not self.is_load_from_checkpoint:
            if self.need_to_load_pre_trained_weights:
                state_dict = self.get_pre_trained_weights()
            elif self.need_to_initialize_weights:
                state_dict = self.construct_hf_model().state_dict()
        
        progress = tqdm(
            desc='Loading weights',
            total=sum(1 for _ in self.parameters()),
            disable=self.global_rank != 0 or self.is_load_from_checkpoint
        )
        module_to_name = {m: n for n, m in self.named_modules()}
        def param_init_fn(module: torch.nn.Module):
            module_name = module_to_name[module]

            if (
                any(t.is_meta for t in module.parameters(recurse=False))
                or any(t.is_meta for t in module.buffers(recurse=False))
            ):
                module.to_empty(device=self.strategy.root_device, recurse=False)
            
            if self.is_load_from_checkpoint:
                return

            for n, p in module.named_parameters(module_name, recurse=False):
                if self.global_rank == 0:
                    p.data.copy_(state_dict[n])
                
                progress.set_postfix_str(n)
                progress.update()
        
        wrap_kwargs = dict(
            param_init_fn=param_init_fn,
            sync_module_states=not self.is_load_from_checkpoint
        )
        if 'auto_wrap_policy' not in self.strategy.kwargs:
            module_classes_to_wrap = set()
            for n in self.hf_model._no_split_modules:
                for m in self.hf_model.modules():
                    if n == m.__class__.__name__:
                        module_classes_to_wrap.add(m.__class__)
                        break
            
            assert len(module_classes_to_wrap) > 0, "`auto_wrap_policy` is not set, trying to infer from `_no_split_modules` but failed."

            wrap_kwargs['auto_wrap_policy'] = ModuleWrapPolicy(module_classes_to_wrap)

        self.hf_model = wrap(self.hf_model, **wrap_kwargs)

        progress.close()

        self.strategy.barrier()

    def __setattr__(self, name: str, value: torch.Tensor | torch.nn.Module) -> None:
        if name == 'peft_model':
            super(torch.nn.Module, self).__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def on_after_configure_model(self) -> None:
        if isinstance(self.strategy, FSDPStrategy):
            self._on_configure_model_end_fsdp()
        else:
            super().on_after_configure_model()

        if self.config.peft_config is not None:
            from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING # type: ignore

            peft_config = self.config.peft_config

            assert not peft_config.is_prompt_learning and not peft_config.is_adaption_prompt
            assert peft_config.peft_type in PEFT_TYPE_TO_TUNER_MAPPING

            self.peft_model = get_peft_model(self, peft_config)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return cross_entropy(
            logits.flatten(end_dim=1),
            labels.flatten(end_dim=1),
            implementation='flash_attn'
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.hf_model(*args, **kwargs).logits

    @copy_method_signature(forward)
    def __call__(): ...

    def training_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> dict[str, Any]:
        labels = shift_labels(batch['labels'])
        position_ids = batch.get('position_ids', None)

        if self.config.neftune_alpha is not None:
            attention_mask = batch['attention_mask']
            inputs_embeds = self.hf_model.get_input_embeddings()(batch['input_ids'])
            noise = torch.zeros_like(inputs_embeds).uniform_(-1, 1)
            input_lengths = torch.sum(attention_mask, 1)
            delta = noise * attention_mask.unsqueeze(2)
            dims = input_lengths * inputs_embeds.size(-1)
            magnitude = self.config.neftune_alpha / torch.sqrt(dims)
            delta = (delta * magnitude.view(-1, 1, 1)).detach()
            inputs_embeds += delta

            logits = self(
                inputs_embeds=inputs_embeds,
                attention_mask=batch['attention_mask'],
                position_ids=position_ids,
                use_cache=False
            )

            self.log('NEFTune Alpha', self.config.neftune_alpha)
        else:
            logits = self(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                position_ids=position_ids,
                use_cache=False
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

        labels = shift_labels(batch['labels'])
        position_ids = batch.get('position_ids', None)
        
        logits = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=position_ids,
            use_cache=False
        )

        loss = self.compute_loss(logits, labels)        
        self.val_perplexity.update(logits, labels)
        self.log('Loss/Val', loss, batch_size=batch_size, sync_dist=True)
    
    def on_validation_epoch_end(self) -> None:
        if getattr(self.strategy, 'tp_size', 1) == 1:
            self.logger.log_metrics({
                'Perplexity/Val': self.val_perplexity.compute()
            })
            self.val_perplexity.reset()

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Any:
        input_ids = batch['input_ids']
        output_ids = self.hf_model.generate(
            input_ids,
            generation_config=self.config.generation_config
        )
        output_ids = output_ids[:, input_ids.size(1):]
        return {
            'output_ids': output_ids
        }
