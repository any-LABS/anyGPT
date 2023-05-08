import argparse
import os
import requests
import tiktoken
import numpy as np

from anyGPT import DEFAULT_DIR

DEFAULT_DATADIR = os.path.join(DEFAULT_DIR, "data")
RAW_DATADIR = os.path.join(DEFAULT_DATADIR, "raw_data")


def _make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anyGPT data preparer",
        description="Downloads and prepares data for training anyGPT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        default="princess_of_mars",
        help="The name of the dataset",
    )
    parser.add_argument(
        "-u",
        "--url",
        dest="url",
        default="https://www.gutenberg.org/cache/epub/62/pg62.txt",
        help="The URL of the dataset pointing to a plain text file.",
    )
    return parser


def _download_data(name, url):
    _make_dir(RAW_DATADIR)
    input_file_path = os.path.join(RAW_DATADIR, f"{name}.txt")
    if not os.path.exists(input_file_path):
        with open(input_file_path, "w") as f:
            f.write(requests.get(url).text)


def _save_to_bin(name, train_ids, val_ids, test_ids):
    dir_name = _make_dir(os.path.join(DEFAULT_DATADIR, name))
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    test_ids = np.array(test_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(dir_name, "train.bin"))
    val_ids.tofile(os.path.join(dir_name, "val.bin"))
    test_ids.tofile(os.path.join(dir_name, "test.bin"))


def _tokenize_data(name, bpe):
    input_file_path = os.path.join(RAW_DATADIR, f"{name}.txt")
    with open(input_file_path, "r") as f:
        data = f.read()
    n = len(data)
    tmp_data = data[: int(n * 0.9)]
    test_data = data[int(n * 0.9) :]
    n = len(tmp_data)
    train_data = tmp_data[: int(n * 0.9)]
    val_data = tmp_data[int(n * 0.9) :]

    enc = tiktoken.get_encoding(bpe)
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    test_ids = enc.encode_ordinary(test_data)
    print(f"Training set has {len(train_ids):,} tokens")
    print(f"Validation set has {len(val_ids):,} tokens")
    print(f"Test set has {len(test_ids):,} tokens")

    _save_to_bin(name, train_ids, val_ids, test_ids)


def prepare_data(name: str, url: str) -> None:
    print(f"Preparing dataset '{name}' from url: {url}.")
    _download_data(name, url)
    _tokenize_data(name, "gpt2")


def main():
    parser = _create_parser()
    args = parser.parse_args()
    prepare_data(args.name, args.url)


if __name__ == "__main__":
    main()
