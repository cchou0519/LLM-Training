# Config

A config file is a YAML file used to set up everything, including seeding, distributed strategy, hyper-parameters, model, data, and more.

When writing your own config file, it's a good idea to refer to some examples in the [config](../config/examples/) directory for guidance.

The config file can be divided into three parts, including trainer, model and data.

Since the `llm-training` command is implemented using [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html#lightning.pytorch.cli.LightningCLI), you can refer to [the lightning tutorials](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html) for more information.

## Trainer

This part is used to set the parameters to be passed to the Lightning Trainer, which controls various general settings, such as distributed strategy, precision, logger, epochs, gradient clipping, checkpointing, and more.

Please refer to the [Trainer API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api) for more information.

```yaml
trainer:
    strategy: ...
    precision: ...
    logger: ...
    max_epochs: 1
    accumulate_grad_batches: 2
    gradient_clip_val: 1.0
    callbacks: ...
```

## Model

This part controls model-related parameters, such as model architecture, pre-trained parameters, optimizer, and more.

First, you need to set the `class_path` to determine which model class to use.

Next, you can use init_args to set initialization parameters.

Specific parameters that can be set depend on the chosen model class.

```yaml
model:
    class_path: path.to.your.model.class
    init_args:
        key1: value1
        key2: value2
```

## Data

This part controls data-related parameters, such as data source, data processing pipeline, batch size, etc.

Similar to the model config, you first use `class_path` to determine the data module class, then use `init_args` to set parameters for it.

```yaml
data:
    class_path: path.to.your.datamodule.class
    init_args:
        key1: value1
        key2: value2
```
