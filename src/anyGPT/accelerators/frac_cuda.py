import logging
from typing import Any, Dict  # noqa
import lightning.pytorch as pl
import torch
from lightning.fabric.accelerators.cuda import num_cuda_devices
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.accelerators.cuda import (
    _check_cuda_matmul_precision,
    _clear_cuda_memory,
)

import os

from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.accelerators import Accelerator

_log = logging.getLogger(__name__)


class FractionalCUDAAccelerator(Accelerator):
    def setup_device(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise MisconfigurationException(
                f"Device should be GPU, got {device} instead."
            )
        _check_cuda_matmul_precision(device)
        torch.cuda.set_device(device)

    def setup(self, trainer: "pl.Trainer") -> None:
        self.set_nvidia_flags(trainer.local_rank)
        _clear_cuda_memory()

    @staticmethod
    def set_nvidia_flags(local_rank: int) -> None:
        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join(str(x) for x in range(num_cuda_devices()))
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        _log.info(f"LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]")

    def teardown(self) -> None:
        _clear_cuda_memory()

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        if isinstance(devices, str):
            devices = eval(devices)
        return list(devices)

    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        return [torch.device("cuda", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        return num_cuda_devices()

    @staticmethod
    def is_available() -> bool:
        return num_cuda_devices() > 0

    @classmethod
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            "fractional_cuda", cls, description=f"{cls.__class__.__name__}"
        )
