# LLM-Training

A distributed training framework for large language models powered by Lightning.

## Supported Training Methods

| Method             | Full Training      | Tensor Parallelism |
| ------------------ | ------------------ | ------------------ |
| Pre-training       | :white_check_mark: | :white_check_mark: |
| Instruction Tuning | :white_check_mark: | :white_check_mark: |
| DPO                | :white_check_mark: | :white_check_mark: |
| ORPO               | :white_check_mark: | :white_check_mark: |

### Pre-training

- Supports Best-fit Bin Packing for less truncation.
- Supports dynamic data sampling via configs, allowing flexible control of data sampling from multiple sources.

### Instruction Tuning

- Supports data packing without cross-contamination.
- Supports NEFTune.

## Supported Models

All GPT-like models supported by HuggingFace are compatible.
However, only text models are supported currently.

Besides, alternative implementations that support additional features for specific model architectures are available.

| Architecture   | Selective Activation Checkpointing | Liger Kernel       | Tensor Parallelism | Sequence Parallelism |
| -------------- | ---------------------------------- | ------------------ | ------------------ | -------------------- |
| LLaMA(2/3/3.x) | :white_check_mark:                 | :white_check_mark: | :white_check_mark: | :white_check_mark:   |
| Phi-3(3.5/4)   | :white_check_mark:                 | :white_check_mark: | :white_check_mark: | :white_check_mark:   |

## Installation

It is recommended to use [conda](https://github.com/conda/conda)/[mamba](https://github.com/mamba-org/mamba) for environment management.

```bash
# Clone this repository
git clone https://github.com/ShinoharaHare/LLM-Training.git && cd LLM-Training

# Optional: Choose the version of LLM Training
# By default, the main branch is used, which includes the latest features and changes but may come with instability.
# Alternatively, you can switch to a specific release from the release page for more stability.
# In most cases, using the latest release is recommended.
git checkout vX.X.X

# Create conda environment
conda env create -f environment.yaml
# or
mamba env create -f environment.yaml

# Activate the created conda environment
conda activate llm-training

# Install LLM Training
./install.sh
```

## Usage

> [!TIP]
> The current documentation is not very comprehensive, as I haven’t had enough time to write it.
> I can only provide brief usage examples, but many details and customizable parameters are not listed or explained in full.
> As a result, you may need to refer to the source code to understand the purpose and usage of some parameters.
> If this does not meet your expectations, you might want to consider using other open-source training frameworks, as there are likely many available in the community.

### Config

To start a training, you will need to write your own config file first.

A config file is a YAML file used to set up everything, including seeding, distributed strategy, hyper-parameters, model, data, and more.

You can refer to the files under the [config](config/examples) directory to write your own config file.

See [document](docs/config.md) for more information.

### Start a training

```bash
llm-training fit --config <CONFIG_PATH>
```

### Multi-node training with SLURM

You can launch a multi-node training using SLURM.

```bash
srun llm-training fit --config <CONFIG_PATH> --trainer.num_nodes <NUM_NODES>
```

See [train.sh](scripts/train.sh) for sbatch script template.

### Convert to Hugging Face

```bash
python scripts/convert_to_hf.py <CKPT_PATH> <OUTPUT_PATH>
```

Note that `<CKPT_PATH>` could either be a file or a folder, depending on the parallelization strategy you are using.
By default, its name will follow this format: `epoch=xxx-step=yyy.ckpt`.

## Hints

### Cross-contamination Attention

To improve training efficiency, we typically perform data packing, where multiple sequences of different lengths are merged into a single sequence, ensuring that each packed sequence has similar lengths.
However, without proper handling, the attention mechanism may focus on irrelevant information, increasing the risk of hallucination in the model.

The model architecture implemented in LLM-Training has already addressed this issue.
On the other hand, if you are using the model architecture provided by HuggingFace, this issue is only handled when Flash Attention 2 is enabled.

Reference: https://github.com/MeetKai/functionary/tree/main/functionary/train/packing

### Faulty Gradient Accumulation

Gradient accumulation is a commonly used technique to simulate large-batch training under limited GPU memory. However, the Unsloth AI team discovered an issue in previous implementations, where the accumulated gradients are inconsistent with those from full-batch training.
The root cause of this problem lies in improper loss normalization, which can also occur in distributed training scenarios.

Currently, LLM-Training has not addressed this issue in its `main` branch, but the `fix-ga-dp` branch includes a fix for `CLM`.
However, our experiments show that the corrected loss calculation does not significantly improve model performance and may even lead to a slight decrease.

If you observe different experimental results, we encourage you to share them.

Reference: https://unsloth.ai/blog/gradient

### Difference between DeepSpeed and FSDP

DeepSpeed and FSDP are both implementations of distributed training, with their algorithms based on ZeRO.
As a result, they are generally considered to deliver similar performance.
However, there are some differences in their details, particularly in parameter precision settings, which are discussed in this [blog](https://huggingface.co/blog/deepspeed-to-fsdp-and-back) post.

In FSDP2's mixed-precision training, we observed that it does not appear to store full-precision parameters separately.
This causes both gradients and optimizer states to remain in half precision, which can significantly degrade training performance.

To address this issue, we implemented an optimizer wrapper that automatically maintains a copy of full-precision parameters.
The optimizer operates on these full-precision parameters and then synchronizes the updates back to the half precision parameters, ensuring training performance.

## Issues

If you encounter any issue while using this framework, please avoid directly contacting the author.
Instead, consider submitting an issue on the repository.
This makes it easier to manage and address errors while also serving as a reference for others who may face the same problem in the future.

Currently, there is no specific format for submitting an Issue. However, when reporting a problem, please provide as much relevant information as possible, such as:

- The version or commit ID of LLM-Training you are using
- The training config file
- The full error message

This will help ensure the issue can be resolved more efficiently.
