# Ubuntu Linux as the base image. You can use any version of Ubuntu here
FROM nvidia/cuda:11.6.0-base-ubuntu20.04
CMD nvidia-smi

# Set UTF-8 encoding
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8


# Install Python

RUN apt-get -y update && \
apt-get -y upgrade

# The following line ensures that the subsequent install doesn't expect user input
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y install python3-pip python3-dev
# RUN apt-get install curl -y

# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"

# Install spaCy
RUN pip3 install --upgrade pip

RUN pip3 install transformers
RUN pip3 install filelock tokenizers huggingface-hub numpy packaging pyyaml regex requests tqdm sacremoses sentencepiece
RUN pip3 install torch
RUN pip3 install datasets
RUN pip3 install numpy
RUN pip3 install nltk
RUN pip3 install protobuf==3.20

RUN pip3 install spacy
RUN python3 -m spacy download en_core_web_lg

# Add the files into container, under QA folder, modify this based on your need
RUN mkdir /QG
ADD ask /QG
ADD a1.txt /QG
ADD src /QG/src
ADD nltk_data /QG/nltk_data
ADD pretrained_model /QG/pretrained_model
ADD trained_model /QG/trained_model

# Change the permissions of programs
CMD ["chmod 777 /QG/*"]

# Set working dir as /QA
WORKDIR /QG
ENTRYPOINT ["/bin/bash", "-c"]