FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        curl \
        git \
        wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --root-user-action ignore --upgrade pip
