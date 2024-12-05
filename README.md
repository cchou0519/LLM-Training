# LLM Training

A LLM training framework powered by Lightning.

## Installation

It is recommended to use [conda](https://github.com/conda/conda)/[mamba](https://github.com/mamba-org/mamba) for environment management.

```bash
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

### Examples

- [Continue Training Phi-3](docs/phi-3_example.md)
