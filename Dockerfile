FROM python:3.9-slim

RUN apt update && apt install -y openssh-server
RUN sed -i 's/#Port 22/Port 2299/' /etc/ssh/sshd_config \
    && sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PermitRootLogin.*$/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 'a\AllowUsers root' /etc/ssh/sshd_config \
    && echo 'root:abc' | chpasswd

COPY requirements.txt ./
RUN pip3 install --no-cache --upgrade -r requirements.txt

CMD service ssh start && sh