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
  openssh-server=7.6p1-4ubuntu0.7 \
  unzip=6.0-21ubuntu1.1 \
  wget=1.19.4-1ubuntu2.2 \
  python3.8=3.8.0-3ubuntu1~18.04.2 \
  python3-dev=3.8.0-3ubuntu1~18.04.2 \
  python3-pip=9.0.1-2.3~ubuntu1.18.04.5 \
  cmake=3.10.2-1ubuntu2.18.04.2 \
  build-essential=12.4ubuntu1 \
  git=1:2.17.1-1ubuntu0.12 \
  libssl-dev=1.1.1-1ubuntu2.1~18.04.20 \
  libx11-dev=2:1.6.4-3ubuntu0.4 \
  xorg-dev=1:7.7+19ubuntu7.1 \
  libglu1-mesa-dev=9.0.0-2.1build1 \
  libusb-1.0-0-dev=2:1.0.21-2 \
  tk=8.6.0+9 && \
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
