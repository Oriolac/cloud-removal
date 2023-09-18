# Use an official Python runtime as a parent image
FROM python:3.8-slim AS laclasse-extraction

# Set the working directory in the container
WORKDIR /app

RUN pip install --trusted-host pypi.python.org pandas argparse

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt

# No need to expose ports or specify CMD as it will be handled by docker-compose
