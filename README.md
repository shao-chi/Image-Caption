# Image Captioning

## Getting Started

### Prerequisites
* Bulid docker
    <br>
    Run `bash docker_run.sh`

* Get Evaluation Function
    ```
    cd core/
    git clone https://github.com/tylin/coco-caption.git

    # Only use 'pycocoevalcap' directory
    mv pycocoevalcap/ metrics/
    ```

* Get YOLOv5 Model
    ```
    cd data/
    git clone https://github.com/ultralytics/yolov5.git
    ```

* Run Stanford NLP Server for tokenizing
    ```
    # Download CoreNLP https://stanfordnlp.github.io/CoreNLP/

    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
        -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
        -status_port 9000 -port 9000 -timeout 15000 & 
    ```

### Preprocess
* Transform images and annotations into specific format
* Split data into training set, validation set, and testing set
    ```
    python3 features.py
    ```

### Train Model
* Train the Transformer model (Caption Generator)
    ```
    python3 main.py train
    ```
    Save model while each epoch was end.

### Evaluation
* Evaluation the dataset
* print the evaluation scores
    ```
    python3 main.py evaluation \
                    --split='test' \
                    --epoch=100 \
                    --beam-size=1
    ```
    `epoch` : which model
    <br>
    `split` : which dataset splited when preprocessing
    <br>
    `beam-size` : beam Search

### Demo
* Generate caption for one image
    ```
    python3 main.py demo \
                    --image-path={IMAGE_PATH} \
                    --epoch=100 \
                    --beam-size=1
    ```

## Reference
* Metrics : https://github.com/tylin/coco-caption
    * BLEU : https://www.aclweb.org/anthology/P02-1040.pdf
    * ROUGE : https://www.aclweb.org/anthology/W04-1013.pdf
    * METEOR : https://www.cs.cmu.edu/~alavie/METEOR/pdf/Banerjee-Lavie-2005-METEOR.pdf
    * CIDEr : https://arxiv.org/pdf/1411.5726.pdf
    * SPICE : https://arxiv.org/pdf/1607.08822.pdf
* Show, Attend, and Tell : https://github.com/yunjey/show-attend-and-tell
* Self Critical : 
    * https://github.com/krasserm/fairseq-image-captioning
    * https://github.com/ruotianluo/ImageCaptioning.pytorch/tree/58b4ea9f77a6b53a1d22138e6502fd4471e5b429
* Deep RL based Image Captioning with Embedding Reward : https://github.com/Pranshu258/Deep_Image_Captioning
* Tokenizer : https://stanfordnlp.github.io/CoreNLP/
* YOLOv5 : https://github.com/ultralytics/yolov5
* ResNet101 : https://arxiv.org/pdf/1512.03385.pdf
* Transformer : https://arxiv.org/pdf/1706.03762.pdf
* Bottom-Up and Top-Down Attention : https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1163.pdf