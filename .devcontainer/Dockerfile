# ベースイメージ
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# パッケージのインストール
RUN apt update && \
    apt install -y sudo python3.10 python3-pip git libgl1 libglib2.0-0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install ipykernel
# ユーザー作成用のビルド引数の追加
ARG USERNAME=ubuntu
ARG GROUPNAME=user
ARG UID=1000
ARG GID=1000
ARG PASSWORD=user

# ユーザー作成
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# requirements.txtをユーザーのホームディレクトリにコピー
USER $USERNAME
COPY requirements.txt /home/$USERNAME/requirements.txt

# パッケージのインストール
RUN pip install -r /home/$USERNAME/requirements.txt