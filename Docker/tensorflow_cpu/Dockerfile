FROM tensorflow/tensorflow:latest

RUN apt-get update && \
    apt-get -y install vim

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install scikit-learn && \
    python3 -m pip install matplotlib

CMD ["bash"]
