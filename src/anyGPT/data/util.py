import os
import pickle

import numpy as np
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
    if isinstance(ints, np.ndarray):
        if len(ints.shape) == 2:
            ints = ints.squeeze()
        ints = ints.tolist()
    return "".join([mapping[i] for i in ints])


def create_enc_dec(dataset):
    meta = load_metadata(dataset)
    if meta is None:
        encoder = tiktoken.get_encoding("gpt2")
        enc = lambda s: encoder.encode(s, allowed_special={"<|endoftext|>"})  # noqa
        dec = lambda l: encoder.decode(l)  # noqa
    else:
        str_to_int, int_to_str = meta["str_to_int"], meta["int_to_str"]
        enc = lambda s: encode(s, str_to_int)  # noqa
        dec = lambda l: decode(l, int_to_str)  # noqa
    return enc, dec


def create_enc_dec_from_metadata(metadata):
    str_to_int, int_to_str = metadata["str_to_int"], metadata["int_to_str"]
    enc = lambda s: encode(s, str_to_int)  # noqa
    dec = lambda l: decode(l, int_to_str)  # noqa
    return enc, dec
