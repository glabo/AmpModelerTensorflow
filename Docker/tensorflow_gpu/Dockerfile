FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Maybe one day we can use the official tensorflow image, but the 5070 is apparently too new
#FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && \
    apt-get -y install vim

RUN python3 -m pip install --upgrade pip && \
    #python3 -m pip install tensorflow[and-cuda] && \
    python3 -m pip install scikit-learn && \
    python3 -m pip install matplotlib

CMD ["bash"]
