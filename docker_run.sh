nvidia-docker build . --tag imagecaptioning/transformer:latest && \
nvidia-docker run \
    -p 6006:6006 \
    --gpus all \
    -it imagecaptioning/transformer:latest \
    /bin/bash