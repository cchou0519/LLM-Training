from contextlib import nullcontext
import logging
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, wrap
from torch.distributed.tensor.parallel import *
from torch.optim import Optimizer
from tqdm.auto import tqdm
from transformers.models.llama import modeling_llama

from llm_training.ops.cross_entropy_op import shift_labels
from llm_training.overrides.strategies import FSDPStrategy
from llm_training.overrides.tensor.parallel import PrepareModuleKeywordInput
from llm_training.utils.decorators import copy_method_signature

from .hf_causal_lm import HFCausalLM
from .patchers.patcher import AutoPatcher

logger = logging.getLogger(__name__)


class HFLlamaForCausalLM(HFCausalLM):
    hf_model_class = modeling_llama.LlamaForCausalLM
    hf_config_class = modeling_llama.LlamaConfig

    def parallelize(self) -> None:
        assert isinstance(self.strategy, FSDPStrategy) and self.strategy.tp_size > 1

        tp_size = self.strategy.tp_size
        tp_mesh = self.strategy.device_mesh['tp']

        parallelize_module(
            self.hf_model,
            tp_mesh,
            {
                'model.embed_tokens': RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1)
                ),
                'model.norm': SequenceParallel(),
                'lm_head': ColwiseParallel(
                    input_layouts=Shard(1),
                    use_local_output=False
                )
            }
        )

        for layer in self.hf_model.model.layers:
            parallelize_module(
                layer,
                tp_mesh,
                {
                    'input_layernorm': SequenceParallel(),
                    'self_attn': PrepareModuleKeywordInput(
                        input_layouts={
                            'hidden_states': Shard(1)
                        },
                        desired_input_layouts={
                            'hidden_states': Replicate()
                        },
                        use_local_output={
                            'hidden_states': False
                        }
                    ),
                    'self_attn.q_proj': ColwiseParallel(),
                    'self_attn.k_proj': ColwiseParallel(),
                    'self_attn.v_proj': ColwiseParallel(),
                    'self_attn.o_proj': RowwiseParallel(output_layouts=Shard(1)),
                    'post_attention_layernorm': SequenceParallel(),
                    'mlp': PrepareModuleInput(
                        input_layouts=Shard(1),
                        desired_input_layouts=Replicate(),
                    ),
                    'mlp.gate_proj': ColwiseParallel(),
                    'mlp.up_proj': ColwiseParallel(),
                    'mlp.down_proj': RowwiseParallel(output_layouts=Shard(1))
                }
            )

            self_attn = layer.self_attn
            self_attn.hidden_size //= tp_size
            self_attn.num_heads //= tp_size
            self_attn.num_key_value_heads //= tp_size

    def configure_model(self) -> None:
        self.hf_model = self.construct_hf_model()
        self._module_to_name = {m: n for n, m in self.named_modules()}

        if getattr(self.strategy, 'tp_size', 1) == 1:
            self.hf_model = AutoPatcher.patch(self.hf_model, self.config)

        if self.global_rank == 0:
            logger.info(f'Config:\n{self.hf_config}')
            logger.info(f'Model:\n{self.hf_model}')
        
        if self.config.enable_gradient_checkpointing:
            self.hf_model.gradient_checkpointing_enable({'use_reentrant': False})

    def on_after_configure_model(self) -> None:        
        strategy = self.strategy
        
        assert isinstance(strategy, FSDPStrategy)

        if strategy.tp_size > 1:
            self.parallelize()

        self.strategy.barrier()

        state_dict = None
        if self.global_rank == 0:
            if self.need_to_load_pre_trained_weights:
                state_dict = self.get_pre_trained_weights()
            elif self.need_to_initialize_weights:
                state_dict = self.construct_hf_model().state_dict()
        
        progress = tqdm(
            desc='Loading weights',
            total=sum(1 for _ in self.parameters()),
            disable=self.global_rank != 0 or not self.need_to_sync_weights
        )

        def param_init_fn(module: torch.nn.Module):
            if (
                any(t.is_meta for t in module.parameters(recurse=False))
                or any(t.is_meta for t in module.buffers(recurse=False))
            ):
                module.to_empty(device=strategy.root_device, recurse=False)
            
            module_name = self._module_to_name.get(module, None)
            assert module_name is not None or module is self.hf_model                

            if not self.need_to_sync_weights or strategy.dp_rank != 0:
                return

            for n, p in module.named_parameters(module_name, recurse=False):
                w = p.data
                placement = None
                if isinstance(p, DTensor):
                    w = p._local_tensor.data
                    assert len(p.placements) == 1
                    placement = p.placements[0]
                
                if isinstance(placement, Shard):
                    scatter_list = None
                    if self.global_rank == 0:
                        shards = state_dict[n].chunk(p._spec.num_shards, dim=placement.dim)
                        scatter_list = [shards[strategy.get_rank(i, 'tp')].to(p) for i in range(strategy.tp_size)]
                    dist.scatter(w, scatter_list, src=0, group=strategy.tp_mesh.get_group())
                elif placement is None or isinstance(placement, Replicate):
                    if self.global_rank == 0:
                        w.copy_(state_dict[n])
                    
                    if strategy.tp_size > 1:
                        dist.broadcast(w, src=0, group=strategy.tp_mesh.get_group())
                else:
                    raise NotImplementedError()
                
                progress.set_postfix_str(n)
                progress.update()

        self.hf_model = wrap(
            self.hf_model,
            auto_wrap_policy=ModuleWrapPolicy({modeling_llama.LlamaDecoderLayer}),
            param_init_fn=param_init_fn,
            sync_module_states=self.need_to_sync_weights
        )

        progress.close()

        self.strategy.barrier()

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if getattr(self.strategy, 'tp_size', 1) > 1:
            with loss_parallel():
                return F.cross_entropy(logits.flatten(end_dim=1), labels.flatten(end_dim=1))
        else:
            return super().compute_loss(logits, labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *args,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
        
        kwargs['input_ids'] = input_ids
        kwargs['attention_mask'] = attention_mask
        kwargs['position_ids'] = position_ids
        kwargs['inputs_embeds'] = inputs_embeds

        return self.hf_model(*args, **kwargs).logits
    
    @copy_method_signature(forward)
    def __call__(): ...

    def training_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> torch.Tensor:
        labels = shift_labels(batch['labels'])
        position_ids = batch.get('position_ids', None)
        
        logits = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=position_ids,
            labels=labels,
            use_cache=False
        )

        loss = self.compute_loss(logits, labels)

        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train/Step', loss)

        if self.grad_norm is not None:
            self.log('Gradient Norm', self.grad_norm)

        if getattr(self.strategy, 'tp_size', 1) == 1:
            self.train_perplexity(logits, labels)
            self.log('Perplexity/Train/Step', self.train_perplexity)
        
        self.consumed_samples.update(labels)
        self.consumed_tokens.update(labels)
        self.logger.log_metrics({
            'Consumed Samples': self.consumed_samples.compute(),
            'Consumed Tokens': self.consumed_tokens.compute()
        })
        return loss

    def backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        context = nullcontext() if getattr(self.strategy, 'tp_size', 1) == 1 else loss_parallel()
        with context:
            return super().backward(loss, *args, **kwargs)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
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
        if getattr(self.strategy, 'tp_size', 1) == 1:
            self.val_perplexity.update(logits, labels)
        self.log('Loss/Val', loss, batch_size=batch_size, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        if getattr(self.strategy, 'tp_size', 1) == 1:
            self.logger.log_metrics({
                'Perplexity/Val': self.val_perplexity.compute()
            })
            self.val_perplexity.reset()

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None
    ) -> None:
        assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
        self._grad_norm = self.hf_model.clip_grad_norm_(gradient_clip_val)
