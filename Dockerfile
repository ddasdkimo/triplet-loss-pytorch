FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
# FROM python:3.7.2-stretch

WORKDIR /app

ADD . /app

# EXPOSE 5000
# RUN pip install -r requirements.txt

# docker build -t raidavid/rai_triplet_loss:v1 ./
# docker push raidavid/rai_triplet_loss:v1
# docker run --gpus all -d -it -p 8193:5000 --name rai_triplet_loss --restart=always raidavid/rai_triplet_loss:v1