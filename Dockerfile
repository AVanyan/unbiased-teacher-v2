#FROM nvidia/cuda:11.0-devel-ubuntu18.04-rc
#FROM nvidia/cuda:10.2-devel-ubuntu18.04
#FROM nvidia/cuda:10.2-base
FROM nvidia/cuda:11.2-base

# Install curl
RUN apt-get update && apt-get install -y curl

# Install python3.6 and pip
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \ 
    python3.8-dev

# Update pip
RUN python3.8 -m pip install --upgrade pip

#symlink for convenience
RUN ln -s /usr/bin/python3.8 /usr/bin/python

#Install other libraries from requirements.txt
COPY requirements.txt /tmp/
RUN cd /tmp/ && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
RUN python3.8 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


RUN useradd --create-home appuser

WORKDIR /home/appuser
COPY ./ /home/appuser/
RUN cd /home/appuser/
RUN mkdir -p /lwll/evaluation
RUN mkdir -p /nonexistent
RUN chmod -R 777 /lwll
RUN chmod -R 777 /home/appuser
RUN chmod -R 777 /nonexistent
USER 65534:65534


ENTRYPOINT python full_detection_tasks.py