# syntax=docker/dockerfile:1.7
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:latest

FROM ${UV_IMAGE} AS uv

FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.11
ARG TORCH_CUDA_ARCH_LIST="8.0;9.0"
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/root/.local/bin:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    UV_PROJECT_ENVIRONMENT=/venvs/retrieve \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server git git-lfs curl wget ca-certificates gnupg \
        build-essential jq rsync less vim tmux htop nvtop zstd unzip procps ripgrep \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_* \
    && mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/; s/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

COPY --from=uv /uv /uvx /usr/local/bin/

RUN curl -fsSL https://claude.ai/install.sh | bash \
    && curl -fsSL https://herdr.dev/install.sh | sh \
    && curl -fsSL -o /usr/local/bin/runpodctl \
        https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
    && chmod +x /usr/local/bin/runpodctl \
    && claude --version && herdr --version && runpodctl version

WORKDIR /opt/retrieve-lock
COPY pyproject.toml uv.lock ./
COPY retrieve/pyproject.toml retrieve/README.md retrieve/
COPY evaluation/pyproject.toml evaluation/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install ${PYTHON_VERSION} \
    && uv sync --frozen --python ${PYTHON_VERSION} --no-install-workspace \
        --all-packages --all-groups --no-install-package silvertorch \
    && MAX_JOBS=${MAX_JOBS} uv sync --frozen --python ${PYTHON_VERSION} --no-install-workspace \
        --all-packages --all-groups --extra official \
    && /venvs/retrieve/bin/python -c "import torch, triton, silvertorch; print(torch.__version__, triton.__version__)"

COPY infra/runpod/rootfs/ /
RUN chmod +x /opt/retrieve-pod/*.sh /usr/local/bin/rp-* \
    && cat /opt/retrieve-pod/bashrc.sh /root/.bashrc > /tmp/bashrc && mv /tmp/bashrc /root/.bashrc \
    && echo '. ~/.bashrc' > /root/.bash_profile

WORKDIR /workspace
EXPOSE 22
CMD ["/opt/retrieve-pod/start.sh"]
