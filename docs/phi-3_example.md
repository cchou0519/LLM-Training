# Example: Continue Training Phi-3

In this example, we will demonstrate how to continue training the [phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) model with dummy data.

The example config file can be found [here](/config/examples/phi-3/phi-3-mini_example.yaml).

## Launch the training

Simply run the following command to continue training phi-3

```bash
llm-training fit -c config/examples/phi-3/phi-3-mini_example.yaml
```

## W&B

In the config file for this example, `WandbLogger` is used to log the training progress, so you will need to create and log into a W&B account.

If you don't have an account yet, you can create one from [W&B](https://wandb.ai).

And make sure to log in using the following command.

```bash
wandb login
```

Alternatively, you can set the `mode` of WandbLogger to `disabled` to disable it.

```yaml
...
  logger:
    class_path: llm_training.lightning.WandbLogger
    init_args:
      ...
      mode: disabled
...
```
