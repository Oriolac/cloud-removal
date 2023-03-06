FROM python:3.9

RUN apt-get update

#install geopandas
RUN apt-get install -y libgdal-dev
RUN apt install -y libspatialindex-dev

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install --upgrade pip


RUN mkdir /app

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY eda /app/eda


