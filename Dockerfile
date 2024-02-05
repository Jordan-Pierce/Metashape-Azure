FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Install pip for python3, mesa deb for metashape, nodejs
RUN apt update && \
    apt install -y \
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

# install metashape python headless 2.0.2
RUN wget http://download.agisoft.com/Metashape-2.0.2-cp37.cp38.cp39.cp310.cp311-abi3-linux_x86_64.whl && \
    python3 -m pip install --user Metashape-2.0.2-cp37.cp38.cp39.cp310.cp311-abi3-linux_x86_64.whl && \
    rm -f Metashape-2.0.2-cp37.cp38.cp39.cp310.cp311-abi3-linux_x86_64.whl \

# expose ports for licensing
EXPOSE 5053 5147

# Copy the Python script and config file
COPY src/SfM.py /home/metashape/
COPY config.ini /home/metashape/
COPY requirements.txt /home/metashape/

# Install dependencies
RUN python3 -m pip install --user -r requirements.txt && \
    rm -f requirements.txt

# Store license key as environmental variable
ENV METASHAPE_LICENSE=METASHAPE_LICENSE

# Specify the default command to run when the container starts
CMD ["python3", "/home/metashape/SfM.py", "--config", "/home/metashape/config.ini"]