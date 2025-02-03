# 1. Use miniconda as the base image
FROM continuumio/miniconda3:latest

# 2. Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Install mamba for faster conda operations
RUN conda install -n base -c conda-forge mamba -y

# 4. Copy environment specification file
COPY environment.yaml /tmp/environment.yaml

# 5. Create conda environment using mamba
RUN mamba env create -f /tmp/environment.yaml \
    && conda clean -afy

# 6. Add conda environment to PATH
ENV PATH="/opt/conda/envs/llm-training/bin:${PATH}"

# 7. Set working directory and copy project files
WORKDIR /workspace/LLM-Training
COPY . .

# 8. Set shell to execute in conda environment
SHELL ["conda", "run", "-n", "llm-training", "/bin/bash", "-c"]

# 9. Install project dependencies
RUN chmod +x install.sh && ./install.sh

# 10. Reset shell and configure automatic environment activation
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate llm-training" >> /root/.bashrc

# 11. Default command to run bash
CMD ["/bin/bash"]
