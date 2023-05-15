import os
import pickle

import tiktoken

from anyGPT import DEFAULT_DATADIR


def load_metadata(dataset_name):
    meta_file = os.path.join(DEFAULT_DATADIR, dataset_name, "meta.pkl")
    load_meta = os.path.exists(meta_file)
    meta = None
    if load_meta:
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
    return meta


def encode(string, mapping):
    return [mapping[ch] for ch in string]


def decode(ints, mapping):
    return "".join([mapping[i] for i in ints])


def create_enc_dec(dataset):
    meta = load_metadata(dataset)
    if meta is None:
        encoder = tiktoken.get_encoding("gpt2")
        enc = lambda s: encoder.encode(s, allowed_special={"<|endoftext|>"})
        dec = lambda l: encoder.decode(l)
    else:
        str_to_int, int_to_str = meta["str_to_int"], meta["int_to_str"]
        enc = lambda s: [str_to_int[c] for c in s]
        dec = lambda l: "".join([int_to_str[i] for i in l])
    return enc, dec
