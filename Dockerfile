FROM python:3.9-slim

COPY requirements.txt ./
RUN pip3 install --no-cache --upgrade -r requirements.txt

CMD sh