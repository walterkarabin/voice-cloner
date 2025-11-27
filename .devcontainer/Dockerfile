# Use a valid CUDA 12.1 runtime image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest
ARG GIT_DELTA_VERSION=0.18.2
ARG ZSH_IN_DOCKER_VERSION=1.2.0
ARG USERNAME=node

# Install dev tools, Node, Python, and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    less \
    git \
    procps \
    fzf \
    zsh \
    man-db \
    unzip \
    gnupg2 \
    gh \
    iptables \
    ipset \
    iproute2 \
    dnsutils \
    jq \
    nano \
    vim \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    libsndfile1 \
    libffi-dev \
    wget \
    curl \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js 20
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y nodejs

# Install Python ML stack (GPU)
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install numpy scipy soundfile transformers TTS

# Create non-root user
RUN useradd -m -s /bin/zsh $USERNAME

# Create workspace and config dirs
RUN mkdir -p /workspace /home/$USERNAME/.claude \
  && chown -R $USERNAME:$USERNAME /workspace /home/$USERNAME/.claude

WORKDIR /workspace
ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

# Install Claude globally as non-root
USER root
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}
USER $USERNAME

# Add python local bin directory
ENV PATH=$PATH:/home/$USERNAME/.local/bin

# Install zsh, fzf, powerlevel10k manually
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git /home/$USERNAME/powerlevel10k && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/powerlevel10k && \
    git clone --depth=1 https://github.com/junegunn/fzf.git /home/$USERNAME/.fzf && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.fzf && \
    /home/$USERNAME/.fzf/install --all

# Persist history (safe location in home)
USER root
RUN mkdir -p /home/$USERNAME/commandhistory && \
    touch /home/$USERNAME/commandhistory/.bash_history && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/commandhistory

USER $USERNAME
ENV PROMPT_COMMAND='history -a'
ENV HISTFILE=/home/$USERNAME/commandhistory/.bash_history


# Firewall script
USER root
COPY init-firewall.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/init-firewall.sh && \
    echo "$USERNAME ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/node-firewall && \
    chmod 0440 /etc/sudoers.d/node-firewall

# Devcontainer labels
LABEL "devcontainer.local_folder"="/workspace" \
      "devcontainer.config_file"="/workspace/.devcontainer/devcontainer.json"

# Set no password for sudo
USER root
RUN echo "node ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/99-node-all && chmod 0440 /etc/sudoers.d/99-node-all
USER $USERNAME
