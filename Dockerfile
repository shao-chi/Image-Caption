FROM nvidia/cuda:10.2-base

WORKDIR /Transformer

ADD . /Transformer

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
    
# Install OpenJDK-11
RUN apt-get -y update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean

RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt -y install libgl1-mesa-glx
RUN apt-get -y install screen

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
