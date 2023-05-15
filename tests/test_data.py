import pytest
from anyGPT.data.next_token_dataset import NextTokenDataset
from anyGPT.data.prepare_data import _tokenize_data
import numpy as np


@pytest.fixture(scope="session")
def dataset_file(tmp_path_factory):
    string = "This is a test. This is a test."
    train_ids, val_ids, test_ids = _tokenize_data(string, "gpt2")
    train_ids = np.array(train_ids, np.uint16)
    filename = tmp_path_factory.mktemp("data") / "train.bin"
    train_ids.tofile(filename)
    return filename


def test_next_token_dataset(dataset_file):
    dataset = NextTokenDataset(dataset_file, "train", 4)
    x, y = dataset[0]
    assert not np.array_equal(x, y)
    for i in range(1, len(x)):
        assert x[i] == y[i - 1]

    assert dataset.data[4] == y[-1]
