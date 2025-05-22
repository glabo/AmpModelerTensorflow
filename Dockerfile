FROM tensorflow/tensorflow:latest

WORKDIR /workspace

RUN echo "DOINGSOMETHING" && \
    apt-get update && \
    apt-get -y install vim

RUN python3 -m pip install --upgrade pip && \
    pip install scikit-learn

CMD ["bash"]