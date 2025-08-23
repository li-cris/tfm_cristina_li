FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    nano \
    wget

# Upgrade pip
RUN pip install --upgrade pip

# Create and set the working directory
RUN mkdir /wdir
WORKDIR /wdir

# Install Python dependencies
COPY requirements.txt /wdir/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /wdir/requirements.txt \
 && rm /wdir/requirements.txt
