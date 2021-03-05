nvidia-docker build . --tag imagecaptioning/transformer:latest && \
nvidia-docker run \
    -p 6006:6006 \
    --gpus all \
    -v /home/shaochi/ImageCaption/Transformer:/Transformer \
    -it imagecaptioning/transformer:latest \
    /bin/bash