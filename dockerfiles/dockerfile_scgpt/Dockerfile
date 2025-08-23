FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime 

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

# Copy the requirements.txt file to the container
COPY requirements.txt /wdir/requirements.txt
COPY no_deps_requirements.txt /wdir/requirements_no_deps.txt

# Install Python dependencies from requirements.txt and requirements_no_deps.txt
RUN pip install -r /wdir/requirements.txt \
    && pip install --no-deps -r /wdir/requirements_no_deps.txt

#Remove requirements
RUN rm /wdir/requirements.txt /wdir/requirements_no_deps.txt