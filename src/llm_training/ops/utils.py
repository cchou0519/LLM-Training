import triton

MAX_FUSED_SIZE = 65536
next_power_of_2 = triton.next_power_of_2

# torch.cuda.amp.custom_fwd is deprecated >= 2.4
import torch
from packaging.version import Version

if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")
pass


import triton
# tl.math.tanh now is libdevice.tanh
from packaging.version import Version

if Version(triton.__version__) >= Version("3.0.0"):
    from triton.language.extra import libdevice
    triton_tanh = libdevice.tanh
else:
    import triton.language as tl
    triton_tanh = tl.math.tanh
pass


def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps
pass
