FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Belgium

RUN apt update && apt install -y \
  openssh-server \
  unzip \
  wget \
  python3.8 \
  python3-dev \
  python3-pip \
  cmake \
  build-essential \
  git \
  libssl-dev \
  libx11-dev \
  xorg-dev \
  libglu1-mesa-dev \
  libusb-1.0-0-dev \
  tk && \
  apt clean


RUN sed -i 's/#Port 22/Port 2299/' /etc/ssh/sshd_config \
    && sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PermitRootLogin.*$/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 'a\AllowUsers root' /etc/ssh/sshd_config \
    && echo 'root:abc' | chpasswd

COPY requirements.txt ./

RUN wget https://github.com/IntelRealSense/librealsense/archive/refs/tags/v2.50.0.zip -O librealsense.zip \
    && unzip librealsense.zip \
    && cd librealsense-2.50.0 \
    && mkdir build && cd build && cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true && make -j4 && make install && export PYTHONPATH=$PYTHONPATH:/usr/local/lib

RUN sed -i "s/pyrealsense/# pyrealsense/" requirements.txt && \
  pip3 install --no-cache --upgrade -r requirements.txt

CMD service ssh start && sh
