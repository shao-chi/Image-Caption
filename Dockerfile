FROM python:3.6

WORKDIR /Transformer

ADD . /Transformer

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]

# ENV NCCL_VERSION 2.7.8

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     cuda-libraries-$CUDA_PKG_VERSION \
#     cuda-npp-$CUDA_PKG_VERSION \
#     cuda-nvtx-$CUDA_PKG_VERSION \
#     libcublas10=10.2.1.243-1 \
#     libnccl2=$NCCL_VERSION-1+cuda10.1 \
#     && apt-mark hold libnccl2 \
#     && rm -rf /var/lib/apt/lists/*

# # apt from auto upgrading the cublas package. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
# RUN apt-mark hold libcublas10

RUN apt update
RUN apt install libgl1-mesa-glx
RUN apt-get install screen