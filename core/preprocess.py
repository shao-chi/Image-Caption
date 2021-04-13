import os

import cv2
import ujson
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from nltk.parse import CoreNLPParser
from collections import Counter
from torchvision import transforms
from torchvision.models import resnet101
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from data.detect_for_preprocess import get_boxes
from data.yolov5.utils.datasets import LoadImages
from core.config import NUM_OBJECT, ENCODE_DIM_FEATURES, ENCODE_DIM_POSITIONS, \
                        IMAGE_MODEL, RESNET_DEVICE, FRCNN_DEVICE

parser = CoreNLPParser(url='http://localhost:9000')
# lemmatizer = WordNetLemmatizer()


class ResnetExtractor(nn.Module):
    def __init__(self):
        super(ResnetExtractor, self).__init__()
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        self.size = 224
        self.tfms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std),])

        submodule = resnet101(pretrained=True)
        modules = list(submodule.children())[:9]

        self.submodule = nn.Sequential(*modules).to(RESNET_DEVICE)

    def forward(self, x):
        x = self.submodule(x.to(RESNET_DEVICE))
        x = x.flatten(1)

        return x.cpu().numpy()

    def transform(self, image):
        image_resized = cv2.resize(image, (self.size, self.size), \
                                  interpolation=cv2.INTER_CUBIC)
        image_resized = self.tfms( \
            Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0)

        return image_resized

    @property
    def transforms(self):
        return self.tfms

    @property
    def image_size(self):
        return self.size


class FasterRCNNExtractor:
    def __init__(self):
        super(FasterRCNNExtractor, self).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.submodule = fasterrcnn_resnet50_fpn(pretrained=True).to(FRCNN_DEVICE)
        self.submodule.eval()

    def get_boxes(self, image_path, num_obj):
        image = self.load_image(image_path)
        features = self.submodule(image.to(FRCNN_DEVICE))[0]

        return features['boxes'].cpu().detach().numpy()[:num_obj], \
                features['scores'].cpu().detach().numpy()[:num_obj], \
                features['labels'].cpu().detach().numpy()[:num_obj]

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = self.transform(image)
        image = image.view(1, *image.shape)

        return image

# TODO: VGG19


def image_feature_YOLOv5(image_path, num_obj=NUM_OBJECT, save_img=False, max_obj=False):
    assert IMAGE_MODEL == 'YOLOv5'

    weights = './data/yolov5/yolov5x.pt'
    model = ResnetExtractor()
    image_size = model.image_size
    transforms = model.transforms
    img_tensor, positions, xyxy = get_boxes(weights,
                                            image_path=image_path,
                                            num_obj=num_obj,
                                            transforms=transforms,
                                            image_size=image_size,
                                            save_img=save_img)

    if max_obj:
        zipper = list(zip(img_tensor.tolist(), positions, xyxy))
        zipper = sorted(zipper, key=lambda z: (z[2][2]-z[2][0]) * (z[2][3]-z[2][1]), reverse=True)[:max_obj]
        img_tensor, positions, xyxy = zip(*zipper)
        positions, xyxy = [positions[0]], [xyxy[0]]
        img_tensor = torch.FloatTensor(img_tensor)

    dataset = LoadImages(image_path, img_size=640)
    for _, _, im0s, _ in dataset:
        im0s_resized = model.transform(image=im0s)

        if len(img_tensor) == 0:
            img_tensor = im0s_resized
        else:
            img_tensor = torch.cat([im0s_resized, img_tensor])

        xyxy_ = [0, 0, 1, 1]
        zeros = [0] * 80
        positions = [xyxy_ + zeros] + positions # xyxy, class * conf

        if len(positions) < num_obj + 1:
            for _ in range(num_obj + 1 - len(positions)):
                positions += [([0] * ENCODE_DIM_POSITIONS)]

        with torch.no_grad():
            features = model(img_tensor)

        if features.shape[0] < num_obj + 1:
            features = np.concatenate([features, \
                    np.zeros((num_obj + 1 - features.shape[0], ENCODE_DIM_FEATURES))])

    return np.array(features).astype(float), \
            np.array(positions).astype(float), \
            np.array(xyxy)


def image_feature_FasterRCNN(image_path, num_obj=NUM_OBJECT, save_img=False):
    assert IMAGE_MODEL == 'FasterRCNN'

    fasterRCNN_model = FasterRCNNExtractor()
    resnet_model = ResnetExtractor()
    boxes, scores, labels = fasterRCNN_model.get_boxes(image_path, num_obj)

    image = cv2.imread(image_path)
    img_tensor = resnet_model.transform(image=image)
    positions = [[0, 0, 1, 1] + [0] * 91]

    if save_img:
        labels_txt = []
        coco_names = open('./data/coco_labels.txt', 'r').read().split('\n')

    for index, (x1, y1, x2, y2) in enumerate(boxes):
        label_id = labels[index] - 1

        position = [y1 / image.shape[0], y2 / image.shape[0],
                    x1 / image.shape[1], x2 / image.shape[1]]
        label = [0] * 91
        label[label_id] = scores[index]
        positions += [position + label]

        obj_image = image[int(y1):int(y2), int(x1):int(x2)]
        try:
            obj_transformed = resnet_model.transform(image=obj_image)
        except:
            continue
        img_tensor = torch.cat([img_tensor, obj_transformed])

        if save_img:
            cv2.rectangle(image, (int(x1), int(y1)),
                          (int(x2), np.int32(y2)), (0, 255, 255), 1, 8, 0)
            label_name = coco_names[label_id]
            font_scale = 0.3
            thickness = 1
            text = f'{label_name} {str(scores[index])[:4]}'
            text_width, text_height = \
                    cv2.getTextSize(text=text,
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale,
                                    thickness=thickness)[0]
            cv2.rectangle(img=image,
                          pt1=(int(x1) - 2, int(y1) + 2),
                          pt2=(int(x1) + text_width + 2, int(y1) - text_height - 2),
                          color=(255, 50, 255),
                          thickness=cv2.FILLED)
            cv2.putText(img=image, text=text, org=(int(x1), int(y1)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        color=(255, 255, 255),
                        thickness=thickness,
                        lineType=cv2.LINE_AA)
            labels_txt.append(f'{label_name} {x1} {y1} {x2} {y2}')

    if save_img:
        name = os.path.basename(image_path)[:-4]
        write_dir = f'./demo/{name}/FasterRCNN'
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        cv2.imwrite(f'{write_dir}/frcnn_{name}.jpg', image)

        labels_txt = '\n'.join(labels_txt)
        with open(f'{write_dir}/labels_{name}.txt', 'w') as txt_file:
            txt_file.write(labels_txt)

    with torch.no_grad():
        features = resnet_model(img_tensor)

    if features.shape[0] < num_obj + 1:
        features = np.concatenate([features, \
                    np.zeros((num_obj + 1 - features.shape[0], ENCODE_DIM_FEATURES))])

    if len(positions) < num_obj + 1:
        for _ in range(num_obj + 1 - len(positions)):
            positions += [([0] * ENCODE_DIM_POSITIONS)]

    return np.array(features).astype(float), \
            np.array(positions).astype(float), \
            boxes


def process_caption_data(caption_file, image_dir, max_length):
    print(f'Processing {caption_file} ...', end=' ')

    with open(caption_file) as f:
        caption_data = ujson.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]}
    id_to_filename = {image['id']: image['file_name'] \
                        for image in caption_data['images']}

    # data is a list of dictionary 
    # which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]
    
    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)
    
    del_index = []
    original_max_length = 0
    for i, caption in enumerate(tqdm(caption_data['caption'])):
        caption = caption.replace('.', '') \
                         .replace(',', '') \
                         .replace("'", "") \
                         .replace('"', '')
        caption = caption.replace('&', 'and') \
                         .replace('(', '') \
                         .replace(")", "") \
                         .replace('-', ' ')

        # tokens = word_tokenize(caption)
        tokens = list(parser.tokenize(caption.lower()))

        caption = " ".join(tokens)  # replace multiple spaces
        caption_data.at[i, 'caption'] = caption
        
        if len(tokens) > original_max_length:
            original_max_length = len(tokens)

        # if len(word_tokenize(caption)) > max_length:
        if len(list(parser.tokenize(caption))) > max_length:
            del_index.append(i)

    print('Original max length: ', original_max_length)
    
    # delete captions if size is larger than max_length
    print(f"The number of captions before deletion: {len(caption_data)}")
    caption_data = caption_data.drop(caption_data.index[del_index])
    caption_data = caption_data.reset_index(drop=True)
    print(f"The number of captions after deletion: {len(caption_data)}")

    return caption_data

def build_vocab(annotations, threshold=1):
    full_vocabulary = Counter()
    max_length = 0

    for caption in annotations['caption']:
        tokens = list(parser.tokenize(caption))
        # for i in range(len(tokens)):
        #     tokens[i] = lemmatizer.lemmatize(tokens[i])

        full_vocabulary.update(tokens)
        
        if len(tokens) > max_length:
            max_length = len(tokens)

    vocab = [word for word in full_vocabulary \
                    if full_vocabulary[word] >= threshold]
    print(f'Filtered {len(full_vocabulary)} words to \
            {len(vocab)} words with word count threshold {threshold}.')

    # word_index = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    word_index = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<UNK>': 3}
    index = 4
    for word in vocab:
        word_index[word] = index
        index += 1

    print("Max length of caption: ", max_length)

    return word_index


def build_caption_vector(annotations, word_index, max_length=24):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples, max_length+2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        # words = word_tokenize(caption)
        words = list(parser.tokenize(caption))

        caption_vector = []
        caption_vector.append(word_index['<START>'])

        for word in words:
            # word = lemmatizer.lemmatize(word)
            if word in word_index:
                caption_vector.append(word_index[word])
            else:
                caption_vector.append(word_index['<UNK>'])

        caption_vector.append(word_index['<END>'])
        
        # pad short caption with the special null token '<NULL>' 
        # to make it fixed-size vector
        if len(caption_vector) < (max_length + 2):
            for _ in range(max_length + 2 - len(caption_vector)):
                caption_vector.append(word_index['<NULL>'])

        assert len(caption_vector) == max_length + 2
        
        captions[i, :] = np.array(caption_vector)

    print("Finished building caption vectors")
    return captions


def build_file_names(annotations):
    image_file_names = []
    id_index = {}
    index = 0

    image_ids = annotations['image_id']
    file_names = annotations['file_name']

    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_index:
            id_index[image_id] = index
            image_file_names.append(file_name)

            index += 1

    file_names = np.array(image_file_names)

    return file_names, id_index


def build_image_indices(annotations, id_index):
    image_indices = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_indices[i] = id_index[image_id]

    return image_indices