#!/bin/bash

# Make sure `ninja` and `packaging` is installed before installing `flash_attn`
pip install ninja packaging
# Force a rebuild of `flash_attn` in case .so files built with an incompatible version of CUDA is cached.

FA_VERSION=$(cat pyproject.toml | grep -oE '"flash-attn[^"]+"')
FA_VERSION=${FA_VERSION:1:-1}
pip install $FA_VERSION --no-build-isolation --no-cache-dir

pip install -e .[deepspeed]
