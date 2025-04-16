FROM ubuntu:22.04

WORKDIR /workspace/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        curl \
        git \
        wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements_other.txt requirement_torch.txt ./

RUN pip install --root-user-action ignore --upgrade pip && \
    pip install --root-user-action ignore --requirement requirements_other.txt && \
    pip install --root-user-action ignore --requirement requirements_torch.txt

RUN rm requirements_other.txt requirements_torch.txt
