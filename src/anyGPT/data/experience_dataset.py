from typing import Callable

from torch.utils.data import IterableDataset


class ExperienceDataset(IterableDataset):
    def __init__(self, collect_experience: Callable):
        self.collect_experience = collect_experience

    def __iter__(self):
        return self.collect_experience()
