FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    apt -y clean && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8

RUN apt update && \
    apt install -y  -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" keyboard-configuration && \
    apt install -y git xterm wget python3-pip python3-opencv && \
    apt -y clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install onnxruntime-gpu

WORKDIR /workdir
RUN git clone https://github.com/Kazuhito00/Informative-Drawings-ONNX-Sample.git
WORKDIR /workdir/Informative-Drawings-ONNX-Sample

# USE Usb Camera
CMD ["bash"]
# ================================
# docker build . -t informative_drawings
# docker run --rm -it -e DISPLAY=$DISPLAY --privileged --device /dev/video0:/dev/video0:mwr --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix informative_drawings python3 sample_onnx.py

