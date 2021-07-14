#!/usr/bin/env bash

sudo apt install -y nano \
                    cuda-toolkit-10-1 \
                    cuda-libraries-10-1 \
                    libcudnn8 \
                    build-essential \
                    libssl-dev \
                    zlib1g-dev \
                    libbz2-dev \
                    libreadline-dev \
                    libsqlite3-dev \
                    wget curl llvm \
                    libncurses5-dev \
                    libncursesw5-dev \
                    xz-utils tk-dev \
                    libffi-dev liblzma-dev \
                    python-openssl git
curl https://pyenv.run | bash
sed -i '1ieval "$(pyenv init --path)"' ~/.profile
sed -i '1iexport PATH="$PYENV_ROOT/bin:$PATH"' ~/.profile
sed -i '1iexport PYENV_ROOT="$HOME/.pyenv"' ~/.profile
sed -i '$ a eval "$(pyenv init -)"' ~/.bashrc
sed -i '$ a export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE=1' ~/.bashrc
eval "$(cat ~/.bashrc | tail -n +10)"
pyenv install 3.8.3
pyenv global 3.8.3
