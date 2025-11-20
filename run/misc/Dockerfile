# syntax=docker/dockerfile:1
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CONDA_ENV_DIR=/opt/conda/envs/.exo_25_RM

# Minimal system deps + build tools for installing local packages (adjust if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    curl \
    bzip2 \
    tar \
    gzip \
    git \
    build-essential \
    python3-dev \
    gfortran \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Build-time args (defaults); can be overridden with --build-arg
ARG NB_USER=worker
ARG NB_UID=1000
ARG NB_GID=1000

# Create group and non-root user
RUN groupadd -g ${NB_GID} ${NB_USER} || true \
 && useradd -m -s /bin/bash -u ${NB_UID} -g ${NB_GID} ${NB_USER}

WORKDIR /home/${NB_USER}

# Copy packed conda env and ellc tarball into the build context root of the image
COPY exo_25_RM_condapack.tar.gz /exo_25_RM_condapack.tar.gz
COPY ellc.tar.gz /ellc.tar.gz

# Unpack the conda-packed environment, run conda-unpack using the env's python if present,
# unpack ellc.tar.gz into /ellc_src and install using the env's python -m pip, then clean up.
RUN mkdir -p ${CONDA_ENV_DIR} \
 && tar -xzf /exo_25_RM_condapack.tar.gz -C ${CONDA_ENV_DIR} \
 && rm /exo_25_RM_condapack.tar.gz \
 && if [ -x "${CONDA_ENV_DIR}/bin/conda-unpack" ]; then "${CONDA_ENV_DIR}/bin/python" "${CONDA_ENV_DIR}/bin/conda-unpack"; fi \
 && mkdir -p /ellc_src \
 && tar -xzf /ellc.tar.gz -C /ellc_src --strip-components=1 \
 && rm /ellc.tar.gz \
 && ${CONDA_ENV_DIR}/bin/python -m pip install --no-deps /ellc_src \
 && rm -rf /ellc_src \
 && chown -R ${NB_USER}:${NB_USER} ${CONDA_ENV_DIR} /home/${NB_USER}

# Make the env's bin first in PATH
ENV PATH="${CONDA_ENV_DIR}/bin:${PATH}"

# Switch to non-root user
USER ${NB_USER}
WORKDIR /home/${NB_USER}

# Default command: interactive shell (HTCondor will override)
CMD ["/bin/bash", "-l"]
