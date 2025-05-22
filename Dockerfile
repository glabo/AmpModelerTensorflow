FROM tensorflow/tensorflow:latest

WORKDIR /workspace

RUN echo "DOINGSOMETHING" && apt-get update && apt-get -y install vim

CMD ["bash"]