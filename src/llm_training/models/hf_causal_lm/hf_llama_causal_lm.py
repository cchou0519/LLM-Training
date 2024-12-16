from functools import partial, wraps

import torch
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,
                                               SequenceParallel,
                                               parallelize_module)
from torch.distributed.tensor.placement_types import Replicate, Shard
from transformers.models.llama.modeling_llama import (LlamaConfig,
                                                      LlamaForCausalLM,
                                                      LlamaModel)

from llm_training.ops.attention_op import prepare_packed_4d_causal_mask

from .hf_causal_lm import HFCausalLM


class HFLlamaCausalLM(HFCausalLM):
    hf_model: LlamaForCausalLM

    hf_model_class = LlamaForCausalLM
    hf_config_class = LlamaConfig

    def configure_tensor_parallel(self, tp_mesh):
        if tp_mesh.size() == 1:
            return

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
                    'self_attn': PrepareModuleInput(
                        input_kwarg_layouts={'hidden_states': Shard(1)},
                        desired_input_kwarg_layouts={'hidden_states': Replicate()}
                    ),
                    'self_attn.q_proj': ColwiseParallel(),
                    'self_attn.k_proj': ColwiseParallel(),
                    'self_attn.v_proj': ColwiseParallel(),
                    'self_attn.o_proj': RowwiseParallel(output_layouts=Shard(1)),
                    'post_attention_layernorm': SequenceParallel(),
                    'mlp': PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),)
                    ),
                    'mlp.gate_proj': ColwiseParallel(),
                    'mlp.up_proj': ColwiseParallel(),
                    'mlp.down_proj': RowwiseParallel(output_layouts=Shard(1))
                }
            )

            self_attn = layer.self_attn
            self_attn.num_heads //= tp_mesh.size()
            self_attn.num_key_value_heads //= tp_mesh.size()

    def configure_fully_sharded_data_parallel(self, dp_mesh, reshard_after_forward, mp_policy, offload_policy, **kwargs):
        if dp_mesh.size() == 1:
            return
        
        fully_shard_ = partial(
            fully_shard,
            mesh=dp_mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy
        )

        # shard the entire model to support gradient clipping.
        # see the following issues:
        # https://github.com/pytorch/pytorch/issues/121020
        # https://github.com/pytorch/pytorch/issues/134212
        
        fully_shard_(self.hf_model.model.embed_tokens)
        fully_shard_(self.hf_model.model.norm)
        fully_shard_(self.hf_model.lm_head)
        
        for layer in self.hf_model.model.layers:
            fully_shard_(layer)


def _patch_packed_attention_mask_for_sdpa() -> None:
    fn = LlamaModel._prepare_4d_causal_attention_mask_with_cache_position

    @staticmethod
    @wraps(fn)
    def patched_fn(attention_mask: torch.Tensor, *args, **kwargs):
        causal_mask = fn(attention_mask, *args, **kwargs)
        causal_mask = prepare_packed_4d_causal_mask(attention_mask, causal_mask, inplace=True)
        return causal_mask
    
    LlamaModel._prepare_4d_causal_attention_mask_with_cache_position = patched_fn


_patch_packed_attention_mask_for_sdpa()
