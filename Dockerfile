FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Belgium

#sed -i 's/^\(deb .*\)$/\1 non-free/' /etc/apt/sources.list
#apt install curl
#curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
#distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
#curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
#  tee /etc/apt/sources.list.d/nvidia-docker.list
#apt install nvidia-

RUN apt update && apt install -y \
  openssh-server=1:8.2p1-4ubuntu0.5 \
  unzip=6.0-25ubuntu1 \
  wget=1.20.3-1ubuntu2 \
  python3=3.8.2-0ubuntu2 \
  python3-dev=3.8.2-0ubuntu2 \
  python3-tk=3.8.10-0ubuntu1~20.04 \
  python3-pip=20.0.2-5ubuntu1.6 \
  cmake=3.16.3-1ubuntu1 \
  build-essential=12.8ubuntu1.1 \
  git=1:2.25.1-1ubuntu3.5 \
  libssl-dev=1.1.1f-1ubuntu2.16 \
  libx11-dev=2:1.6.9-2ubuntu1.2 \
  xorg-dev=1:7.7+19ubuntu14 \
  libglu1-mesa-dev=9.0.1-1build1 \
  libusb-1.0-0-dev=2:1.0.23-2build1 \
  tk=8.6.9+1 && \
  apt clean


RUN sed -i 's/#Port 22/Port 2299/' /etc/ssh/sshd_config \
    && sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PermitRootLogin.*$/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#X11UseLocalhost.*$/X11UseLocalhost no/' /etc/ssh/sshd_config \
    && sed -i 's/#AddressFamily.*$/AddressFamily inet/' /etc/ssh/sshd_config \
    && echo 'AllowUsers root' >> /etc/ssh/sshd_config \
    && passwd -d root

COPY requirements.txt ./

RUN wget https://github.com/IntelRealSense/librealsense/archive/refs/tags/v2.50.0.zip -O librealsense.zip \
    && unzip librealsense.zip \
    && cd librealsense-2.50.0 \
    && mkdir build && cd build && cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true && make -j4 && make install && export PYTHONPATH=$PYTHONPATH:/usr/local/lib

RUN sed -i "s/pyrealsense/# pyrealsense/" requirements.txt && \
  pip3 install --no-cache --upgrade -r requirements.txt

CMD service ssh start && bash
