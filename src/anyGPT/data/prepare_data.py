import argparse
import os
import pickle

import numpy as np
import requests
import tiktoken

from anyGPT import RAW_DATADIR, DEFAULT_DATADIR
from anyGPT.data.util import encode


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
    parser.add_argument(
        "-c",
        "--char",
        dest="is_char",
        action="store_true",
        default=False,
        help="Encode as character tokens instead of word tokens.",
    )
    return parser


def _download_data(name, url):
    _make_dir(RAW_DATADIR)
    input_file_path = os.path.join(RAW_DATADIR, f"{name}.txt")
    if not os.path.exists(input_file_path):
        with open(input_file_path, "w") as f:
            f.write(requests.get(url).text)


def _save_to_bin(name, train_ids, val_ids, test_ids, meta=None):
    dir_name = _make_dir(os.path.join(DEFAULT_DATADIR, name))
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    test_ids = np.array(test_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(dir_name, "train.bin"))
    val_ids.tofile(os.path.join(dir_name, "val.bin"))
    test_ids.tofile(os.path.join(dir_name, "test.bin"))
    if meta is not None:
        with open(os.path.join(dir_name, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)


def _tokenize_data_char(name):
    input_file_path = os.path.join(RAW_DATADIR, f"{name}.txt")
    with open(input_file_path, "r") as f:
        data = f.read()
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    str_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_str = {i: ch for i, ch in enumerate(chars)}

    n = len(data)
    tmp_data = data[: int(n * 0.9)]
    test_data = data[int(n * 0.9) :]
    n = len(tmp_data)

    train_data = tmp_data[: int(n * 0.9)]
    val_data = tmp_data[int(n * 0.9) :]

    train_ids = encode(train_data, str_to_int)
    val_ids = encode(val_data, str_to_int)
    test_ids = encode(test_data, str_to_int)
    print(f"Character level encoding with {vocab_size} token vocab size.")
    print(f"Unique characters: {''.join(chars)}")
    print(f"Training set has {len(train_ids):,} tokens")
    print(f"Validation set has {len(val_ids):,} tokens")
    print(f"Test set has {len(test_ids):,} tokens")

    meta = {
        "vocab_size": vocab_size,
        "int_to_str": int_to_str,
        "str_to_int": str_to_int,
    }

    _save_to_bin(name, train_ids, val_ids, test_ids, meta)


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


def prepare_data(name: str, url: str, is_char: bool) -> None:
    print(f"Preparing dataset '{name}' from url: {url}.")
    _download_data(name, url)
    if is_char:
        _tokenize_data_char(name)
    else:
        _tokenize_data(name, "gpt2")


def main():
    parser = _create_parser()
    args = parser.parse_args()
    prepare_data(args.name, args.url, args.is_char)


if __name__ == "__main__":
    main()
