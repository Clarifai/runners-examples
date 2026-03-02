# syntax=docker/dockerfile:1.13-labs
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04@sha256:c900a2ac7f3d860c854f7364eadfe32b8cd33933a838f96b71e9bb6525f77a8c

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11
ENV PYTORCH_VERSION=2.8.0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
# The official PyTorch wheels support both AMD64 and ARM64
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

COPY --link=true requirements.txt /home/nonroot/requirements.txt

#RUN ["pip", "install", "--no-cache-dir", "-r", "/home/nonroot/requirements.txt"]
RUN ["pip", "install", "--no-cache-dir", "-r", "/home/nonroot/requirements.txt"]
RUN ["pip", "show", "--no-cache-dir", "clarifai"]

# Set the NUMBA cache dir to /tmp
# Set the TORCHINDUCTOR cache dir to /tmp
# The CLARIFAI* will be set by the templaing system.
ENV NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
    HOME=/tmp \
    DEBIAN_FRONTEND=noninteractive

#####
# Copy the files needed to download
#####
# This creates the directory that HF downloader will populate and with nonroot:nonroot permissions up.
#COPY --chown=nonroot:nonroot downloader/unused.yaml /home/nonroot/main/1/checkpoints/.cache/unused.yaml

#####
# Download checkpoints if config.yaml has checkpoints.when = "build"
COPY --link=true config.yaml /home/nonroot/main/
RUN ["python", "-m", "clarifai.cli", "model", "download-checkpoints", "/home/nonroot/main", "--out_path", "/home/nonroot/main/1/checkpoints", "--stage", "build"]
#####

# Copy in the actual files like config.yaml, requirements.txt, and most importantly 1/model.py
# for the actual model.
# If checkpoints aren't downloaded since a checkpoints: block is not provided, then they will
# be in the build context and copied here as well.
COPY --link=true 1 /home/nonroot/main/1
# At this point we only need these for validation in the SDK.
COPY --link=true requirements.txt config.yaml /home/nonroot/main/

# Add the model directory to the python path.
ENV PYTHONPATH=${PYTHONPATH}:/home/nonroot/main \
    CLARIFAI_PAT=${CLARIFAI_PAT} \
    CLARIFAI_USER_ID=${CLARIFAI_USER_ID} \
    CLARIFAI_RUNNER_ID=${CLARIFAI_RUNNER_ID} \
    CLARIFAI_NODEPOOL_ID=${CLARIFAI_NODEPOOL_ID} \
    CLARIFAI_COMPUTE_CLUSTER_ID=${CLARIFAI_COMPUTE_CLUSTER_ID} \
    CLARIFAI_API_BASE=${CLARIFAI_API_BASE:-https://api.clarifai.com}

USER root
RUN echo "nonroot:x:65532:65532:nonroot user:/home/nonroot:/sbin/nologin" >> /etc/passwd
USER nonroot
    

# Finally run the clarifai entrypoint to start the runner loop and local runner server.
ENTRYPOINT ["python", "-m", "clarifai.runners.server"]
CMD ["--model_path", "/home/nonroot/main"]
#############################
