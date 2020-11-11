FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
MAINTAINER TIB-Visual-Analytics
RUN DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install libsm6 libxext6 libxrender-dev -y
RUN /opt/conda/bin/conda install -y imageio=2.6.1
RUN /opt/conda/bin/conda install -y tqdm
RUN /opt/conda/bin/conda install -y scikit-learn=0.22.1
RUN /opt/conda/bin/conda install -y networkx==2.4