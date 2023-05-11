import os
import pickle

from anyGPT.data.prepare_data import DEFAULT_DATADIR


def load_metadata(dataset_name):
    meta_file = os.path.join(DEFAULT_DATADIR, dataset_name, "meta.pkl")
    load_meta = os.path.exists(meta_file)
    meta = None
    if load_meta:
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
    return meta
