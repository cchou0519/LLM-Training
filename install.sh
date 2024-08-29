#!/bin/bash

# Make sure `packaging` is installed before installing `flash_attn`
pip install packaging
# Force a rebuild of `flash_attn` in case .so files built with an incompatible version of CUDA is cached.
pip install flash_attn==2.5.9.post1 --no-build-isolation --no-cache-dir

pip install -e .[deepspeed]
