# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
FROM python:3.7.2-stretch

WORKDIR /app

ADD . /app

# RUN pip install -r requirements.txt

