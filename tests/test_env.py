import time

import numpy as np

import anyGPT
import gymnasium as gym
import pytest

test_data = [
    (
        "j-hartmann/emotion-english-distilroberta-base",
        "shakespeare_karpathy",
        False,
        "neutral",
    ),
    (
        "madhurjindal/autonlp-Gibberish-Detector-492513457",
        "shakespeare_karpathy",
        False,
        "clean",
    ),
    (
        "j-hartmann/emotion-english-distilroberta-base",
        "shakespeare_karpathy_char",
        False,
        "neutral",
    ),
    (
        "madhurjindal/autonlp-Gibberish-Detector-492513457",
        "shakespeare_karpathy_char",
        False,
        "clean",
    ),
    (
        "j-hartmann/emotion-english-distilroberta-base",
        "shakespeare_karpathy",
        True,
        "neutral",
    ),
    (
        "madhurjindal/autonlp-Gibberish-Detector-492513457",
        "shakespeare_karpathy",
        True,
        "clean",
    ),
    (
        "j-hartmann/emotion-english-distilroberta-base",
        "shakespeare_karpathy_char",
        True,
        "neutral",
    ),
    (
        "madhurjindal/autonlp-Gibberish-Detector-492513457",
        "shakespeare_karpathy_char",
        True,
        "clean",
    ),
]


@pytest.mark.parametrize("model_name,dataset,encoded,label", test_data)
def test_env(model_name, dataset, encoded, label):
    env = gym.make(
        "anyGPT/SequenceClassificationEnv-v0",
        model_name=model_name,
        dataset=dataset,
        encoded=encoded,
        label=label,
    )
    obs, info = env.reset()
    no_gibberish = "Set forth ye, for the clouds come hither."
    gibberish = "Bazinga!"
    if not encoded:
        assert isinstance(obs, str)
    else:
        assert isinstance(obs, np.ndarray)
        no_gibberish = np.array(env.encode(no_gibberish))
        gibberish = np.array(env.encode(gibberish))

    obs, reward, terminated, _, info = env.step(no_gibberish)
    assert obs is None
    assert reward > 0.5
    assert terminated

    obs, reward, terminated, _, info = env.step(gibberish)
    assert obs is None
    assert reward < 0.5
    assert terminated


# def test_emotion_env():
#     env = gym.make(
#         "anyGPT/SequenceClassificationEnv-v0",
#         dataset="shakespeare_karpathy",
#         model_name="j-hartmann/emotion-english-distilroberta-base",
#         label="neutral",
#         encoded=False
#     )
#     obs, info = env.reset()
#     assert isinstance(obs, str)
#     neutral = "Set forth ye, for the clouds come hither."
#     start = time.time()
#     obs, reward, terminated, _, info = env.step(neutral)
#     print(time.time() - start)
#     assert obs is None
#     assert reward > 0.5
#     assert terminated
#     excited = "Bazinga!"
#     obs, reward, terminated, _, info = env.step(excited)
#     assert obs is None
#     assert reward < 0.5
#     assert terminated
