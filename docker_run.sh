docker build . --tag imagecaptioning/transformer:latest && \
docker run --gpus all -it \
    imagecaptioning/transformer:latest /bin/bash