FROM tensorflow/tensorflow:latest-gpu
#FROM tensorflow/tensorflow:2.14.0-gpu

RUN apt-get update && \
    apt-get -y install vim

RUN python3 -m pip install --upgrade pip && \
    #python3 -m pip install tensorflow[and-cuda] && \
    python3 -m pip install scikit-learn && \
    python3 -m pip install matplotlib

CMD ["bash"]
