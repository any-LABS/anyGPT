# Training

Create a config file. In this example, I'll call it `gpt-2-30M.yaml`. You can also check out the [example configuration files][example-configs].

```yaml title="gpt-2-30M.yaml"
model_config:
  name: 'gpt-2-30M'
  block_size: 256
  dropout: 0.2
  embedding_size: 384
  num_heads: 6
  num_layers: 6

training_config:
  learning_rate: 1.0e-3
  batch_size: 8
  accumulate_gradients: 8
  beta2: 0.99
  min_lr: 1.0e-4
  max_steps: 5000
  val_check_interval: 200
  limit_val_batches: 100

io_config:
  experiment_name: 'gpt-2'
  dataset: 'shakespeare_complete'
```

```shell
$ anygpt-train gpt-2-30M.yaml
```