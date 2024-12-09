FROM ubuntu:22.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        default-jre \
        git \
        python3 \
        python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install gdown.
RUN pip install gdown

# Copy the install_compass.sh script into the container, make it executable, and run it
# to install Compass.
COPY install_compass.sh /usr/local/bin/install_compass.sh
RUN chmod u+x /usr/local/bin/install_compass.sh && /usr/local/bin/install_compass.sh
