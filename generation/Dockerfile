#FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
FROM  pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="oriol.alas@eurecat.org"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y libopenmpi-dev libgl1-mesa-glx

RUN apt-get install -yq libgtk2.0-dev

RUN python -m ensurepip --upgrade

COPY requirements.txt .

# RUN python -m pip install torch torchvision torchaudio

RUN python -m pip install -r requirements.txt

COPY . .

