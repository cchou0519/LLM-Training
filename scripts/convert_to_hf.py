import os
from pathlib import Path
from typing import Any

import fire
import torch
import yaml
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.cli import LightningArgumentParser
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from llm_training.models.hf_compat_model import HFCompatModel


def main(
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    config_path: str | None = None,
    eos_tokens: list[str] | None = (
        None
        # ['<|endoftext|>', '<|im_end|>'] # phi-2
        # ['<|end_of_text|>', '<|eot_id|>'] # llama-3
        # ['<|endoftext|>', '<|assistant|>', '<|end|>'] # phi-3
    )
) -> None:
    checkpoint_path = Path(checkpoint_path)

    if output_dir is None:
        output_dir = checkpoint_path.parent / 'hf' / checkpoint_path.stem
    else:
        output_dir = Path(output_dir)

    print('Converting checkpoint')
    checkpoint = convert_checkpoint(checkpoint_path)

    if config_path is not None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint['config']
    
    model, datamodule = instantiate_model_and_datamodule(config)
    
    assert isinstance(model, HFCompatModel), f"{model.__class__} is not supported to be converted to HF version."

    print('Constructing empty HF model')
    hf_model = construct_hf_model(model.hf_model_class, model.load_hf_config())
    
    print('Converting state dict')
    state_dict = checkpoint['state_dict']
    state_dict = model.convert_state_dict_to_hf(state_dict)
    ref_state_dict = hf_model.state_dict()
    hf_model.load_state_dict({k: v for k, v in state_dict.items() if k in ref_state_dict})
    
    if hasattr(datamodule, 'config') and hasattr(datamodule.config, 'tokenizer'):
        tokenizer = datamodule.config.tokenizer
    else:
        tokenizer = model.load_hf_tokenizer()
    
    if hasattr(datamodule, 'config') and hasattr(datamodule.config, 'chat_template'):
        tokenizer.chat_template = datamodule.config.chat_template
    
    if hasattr(datamodule, 'config') and hasattr(datamodule.config, 'max_length'):
        tokenizer.model_max_length = max(tokenizer.model_max_length, datamodule.config.max_length)
    
    if eos_tokens is not None:
        vocab = tokenizer.get_vocab()
        hf_model.generation_config.eos_token_id = [vocab[e] for e in eos_tokens]
        tokenizer.eos_token = eos_tokens[0]
    
    print('Saving model')
    hf_model.save_pretrained(output_dir)
    
    print('Saving tokenizer')
    tokenizer.save_pretrained(output_dir)


def convert_checkpoint(path: Path) -> dict[str, Any]:
    if (
        path.is_dir()
        and path.joinpath('checkpoint').is_dir()
        and path.joinpath('latest').is_file()
        and path.joinpath('zero_to_fp32.py').is_file()
    ):
        print('DeepSpeed checkpoint detected')
        from lightning.pytorch.utilities.deepspeed import \
            convert_zero_checkpoint_to_fp32_state_dict
        return convert_zero_checkpoint_to_fp32_state_dict(path, os.devnull)
    
    if (
        path.is_dir()
        and path.joinpath('meta.pt').is_file()
        and len(list(path.glob('*.distcp'))) > 0
    ):        
        print('FSDP checkpoint detected')
        return convert_fsdp_checkpoint(path)

    return torch.load(path, 'cpu')


def convert_fsdp_checkpoint(path: Path) -> dict[str, Any]:
    from lightning.fabric.utilities.load import (_METADATA_FILENAME,
                                                 _unflatten_dict)
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.metadata import (BytesStorageMetadata,
                                                       TensorStorageMetadata)
    
    reader = FileSystemReader(path)
    metadata = reader.read_metadata()

    tensor_names = [n for n in metadata.state_dict_metadata.keys() if n.startswith('model.')]
    state_dict = {}
    for tensor_name in tensor_names:
        sd_metadata = metadata.state_dict_metadata[tensor_name]

        if isinstance(sd_metadata, BytesStorageMetadata):
            state_dict[tensor_name] = '<bytes_io>'
        elif isinstance(sd_metadata, TensorStorageMetadata):
            state_dict[tensor_name] = torch.empty(
                size=sd_metadata.size,
                dtype=sd_metadata.properties.dtype,
                device=torch.device('cpu'),
                memory_format=sd_metadata.properties.memory_format,
                layout=sd_metadata.properties.layout,
                requires_grad=sd_metadata.properties.requires_grad,
                pin_memory=sd_metadata.properties.pin_memory
            )
        else:
            raise NotImplementedError()

    load(state_dict=state_dict, storage_reader=reader)
    key_map = {n: metadata.planner_data[n] for n in tensor_names}
    state_dict = _unflatten_dict(state_dict, key_map=key_map)['model']
    
    checkpoint = {'state_dict': state_dict}
    # This is the extra file saved by Fabric, with user data separate from weights and optimizer states
    extra_file = path / _METADATA_FILENAME
    extra = torch.load(extra_file, map_location='cpu') if extra_file.is_file() else {}
    checkpoint.update(extra)

    return checkpoint


def instantiate_model_and_datamodule(config: dict[str, Any]) -> tuple[LightningModule, LightningDataModule]:
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(LightningModule, 'model', subclass_mode=True)
    parser.add_lightning_class_args(LightningDataModule, 'data', subclass_mode=True)    
    classes = parser.instantiate_classes({k: config[k] for k in ['model', 'data']})
    return classes['model'], classes['data']


def construct_hf_model(
    hf_model_class: type[_BaseAutoModelClass] | type[PreTrainedModel],
    hf_config: PretrainedConfig
) -> PreTrainedModel:
    current_dtype = torch.get_default_dtype()
    target_dtype = hf_config.torch_dtype
    if isinstance(target_dtype, str):
        target_dtype = getattr(torch, target_dtype, target_dtype)
    torch.set_default_dtype(target_dtype)
    with no_init_weights():
        if issubclass(hf_model_class, _BaseAutoModelClass):
            model = hf_model_class.from_config(hf_config)
        else:
            model = hf_model_class(hf_config)
    torch.set_default_dtype(current_dtype)
    return model


if __name__ == '__main__':
    fire.Fire(main)
