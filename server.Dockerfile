FROM python:3.7

# FROM nvidia/cuda:11.0-runtime-ubuntu18.04 
# RUN apt-get update
# RUN apt-get install python3.6 -y
# RUN apt-get update && apt-get upgrade -y
# RUN apt-get install python3-pip -y 

ARG FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY server.requirements.txt .  
COPY run.sh .
COPY app/backend .

RUN pip3 install --upgrade pip

# RUN apt-get update
# RUN apt-get install git -y

RUN pip3 install -r server.requirements.txt 
# RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

CMD bash /app/run.sh
