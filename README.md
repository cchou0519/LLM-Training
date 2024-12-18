# LLM Training

A LLM training framework powered by Lightning.

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

## Issues

If you encounter any issue while using this framework, please avoid directly contacting the author.
Instead, consider submitting an issue on the repository.
This makes it easier to manage and address errors while also serving as a reference for others who may face the same problem in the future.

Currently, there is no specific format for submitting an Issue. However, when reporting a problem, please provide as much relevant information as possible, such as:

- The version or commit ID of LLM-Training you are using
- The training config file
- The full error message

This will help ensure the issue can be resolved more efficiently.
