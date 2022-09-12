FROM python:3.9-slim

RUN apt update && apt install -y \
  openssh-server=1:8.4p1-5+deb11u1 \
  unzip=6.0-26+deb11u1 \
  wget=1.21-1+deb11u1 \
  python3-dev=3.9.2-3 \
  cmake=3.18.4-2+deb11u1 \
  build-essential=12.9 \
  git=1:2.30.2-1 \
  libssl-dev=1.1.1n-0+deb11u3 \
  libx11-dev=2:1.7.2-1 \
  xorg-dev=1:7.7+22 \
  libglu1-mesa-dev=9.0.1-1 \
  libusb-1.0-0-dev=2:1.0.24-3 \
  tk=8.6.11+1 && \
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
