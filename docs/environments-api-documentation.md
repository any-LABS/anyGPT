# Table of Contents

* [anyGPT.environments.sequence\_classification\_env](#anyGPT.environments.sequence_classification_env)
  * [SequenceClassificationEnv](#anyGPT.environments.sequence_classification_env.SequenceClassificationEnv)
    * [\_\_init\_\_](#anyGPT.environments.sequence_classification_env.SequenceClassificationEnv.__init__)

<a id="anyGPT.environments.sequence_classification_env"></a>

# anyGPT.environments.sequence\_classification\_env

<a id="anyGPT.environments.sequence_classification_env.SequenceClassificationEnv"></a>

## SequenceClassificationEnv Objects

```python
class SequenceClassificationEnv(gym.Env)
```

<a id="anyGPT.environments.sequence_classification_env.SequenceClassificationEnv.__init__"></a>

#### \_\_init\_\_

```python
def __init__(
        dataset: str = None,
        render_mode: str = None,
        block_size: int = 2,
        label: str = "clean",
        model_name: str = "madhurjindal/autonlp-Gibberish-Detector-492513457",
        encoded: bool = True,
        device: str = "cpu")
```

**Arguments**:

- `dataset`: The dataset to sample observations from.
- `render_mode`:
- `block_size`: The size of the observation string.
- `label`: The label to optimize for. Will depend on the HF model loaded.
- `model_name`: The name of the HF sequence classifier.
- `encoded`: If set to true, observations and actions should be encoded with the proper gpt style tokenizer.
