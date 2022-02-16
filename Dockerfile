#FROM nvidia/cuda:10.0-cudnn7-devel
#FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
EXPOSE 3000 5000  8625 8265 


RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get -y install build-essential
RUN apt-get install -y unzip
RUN apt-get install -y wget
RUN apt-get install -y vim
RUN apt update

RUN apt-get install -y python3-pip
RUN apt-get -y install python3-setuptools
RUN apt-get -y install tmux
RUN apt-get -y install git
RUN apt-get -y install cmake
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev 
RUN apt-get install lsof

COPY ./requirements.txt /home
COPY ./baselines /home
WORKDIR /home
RUN ls /home
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  
RUN pip3 install -v torch==1.7.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install google-auth --upgrade  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install ray --upgrade  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -e baselines -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install ray==1.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple