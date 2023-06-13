from lightning.fabric.accelerators.registry import call_register_accelerators
from lightning.pytorch.accelerators import AcceleratorRegistry

from anyGPT.accelerators.frac_cuda import FractionalCUDAAccelerator  # noqa

CUSTOM_ACCELERATORS_BASE_MODULE = "anyGPT.accelerators"
call_register_accelerators(AcceleratorRegistry, CUSTOM_ACCELERATORS_BASE_MODULE)
