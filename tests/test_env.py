import gymnasium as gym
import anyGPT


def test_gibberish_env():
    env = gym.make(
        "anyGPT/SequenceClassificationEnv-v0",
        dataset="shakespeare_karpathy",
        block_size=10,
    )
    obs, info = env.reset()
    assert isinstance(obs, str)
    no_gibberish = "Set forth ye, for the clouds come hither."
    obs, reward, terminated, _, info = env.step(no_gibberish)
    assert obs is None
    assert reward > 0.5
    assert terminated
    gibberish = "Bazinga!"
    obs, reward, terminated, _, info = env.step(gibberish)
    assert obs is None
    assert reward < 0.5
    assert terminated


def test_emotion_env():
    env = gym.make(
        "anyGPT/SequenceClassificationEnv-v0",
        dataset="shakespeare_karpathy",
        block_size=10,
        model_name="j-hartmann/emotion-english-distilroberta-base",
        label="neutral",
    )
    obs, info = env.reset()
    assert isinstance(obs, str)
    neutral = "Set forth ye, for the clouds come hither."
    obs, reward, terminated, _, info = env.step(neutral)
    assert obs is None
    assert reward > 0.5
    assert terminated
    excited = "Bazinga!"
    obs, reward, terminated, _, info = env.step(excited)
    assert obs is None
    assert reward < 0.5
    assert terminated
