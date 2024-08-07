FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04

# Install pip for python3, mesa deb for metashape, nodejs
RUN apt update && \
    apt install -y \
        python3.8 \
        python3-pip \
        wget \
        libglu1-mesa \
        libgl1-mesa-glx \
        libcurl4 \
    && rm -rf /var/lib/apt/lists/*

# add metashape user and swith to it
RUN adduser --home /home/metashape --shell /bin/bash --disabled-password metashape
USER metashape

# set The workdir
WORKDIR /home/metashape

# install metashape python headless 2.1.2
RUN wget http://download.agisoft.com/Metashape-2.1.2-cp37.cp38.cp39.cp310.cp311-abi3-linux_x86_64.whl -P /home/metashape

# Copy requirements and install
COPY requirements.txt /home/metashape/

# Install dependencies
RUN python3 -m pip install --user -r requirements.txt && \
    rm -f requirements.txt && \
    rm -f Metashape-2.1.2-cp37.cp38.cp39.cp310.cp311-abi3-linux_x86_64.whl && \
    pip list

# expose ports for licensing
EXPOSE 5053 5147

# Copy the Python script and config file
COPY src/SfM.py /home/metashape/

# Copy over the input data
COPY input /home/metashape/input/
