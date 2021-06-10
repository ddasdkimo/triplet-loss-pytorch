FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

WORKDIR /app

ADD . /app

# RUN pip install -r requirements.txt

