import os
import logging

import cv2
import ujson
import hickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import gluonnlp as nlp
import mxnet as mx
from nltk.parse import CoreNLPParser
from collections import Counter
from torchvision import transforms
from torchvision.models import resnet101
from torch.utils.data.dataloader import DataLoader

from data.detect_for_preprocess import get_boxes
from data.yolov5.utils.datasets import LoadImages
from core.utils import *
from core.settings import MAX_LENGTH, WORD_COUNT_THRESHOLD, DATA_PATH, NUM_OBJECT


parser = CoreNLPParser(url='http://localhost:9000')
# lemmatizer = WordNetLemmatizer()

class ResnetExtractor(nn.Module):
    def __init__(self):
        super(ResnetExtractor, self).__init__()
        submodule = resnet101(pretrained=True)
        # self.extracted_layer = extracted_layer
        modules = list(submodule.children())[:9]
        # print(modules)

        self.submodule = nn.Sequential(*modules)

    def forward(self, x):
        x = self.submodule(x)
        x = x.flatten(1)
        return x


def image_feature(image_path, model, transforms, image_size,
                  num_obj=NUM_OBJECT, save_img=False):
    weights = './data/yolov5/yolov5x.pt'
    img_tensor, positions, xyxy = get_boxes(weights,
                                            image_path=image_path,
                                            num_obj=num_obj*2,
                                            transforms=transforms,
                                            image_size=image_size,
                                            save_img=save_img)

    all_features = []
    all_positions = []

    dataset = LoadImages(image_path, img_size=640)
    for _, _, im0s, _ in dataset:
        im0s_resized = cv2.resize(im0s, (image_size, image_size), \
                                  interpolation=cv2.INTER_CUBIC)
        im0s_resized = transforms( \
            Image.fromarray(cv2.cvtColor(im0s_resized, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0)

        # img_tensor = im0s_resized
        img_tensor = torch.cat([im0s_resized, img_tensor])
        xyxy_ = [0, 0, 1, 1]
        zeros = [0] * 80
        positions = [xyxy_ + zeros] + positions # xyxy, class * conf

        with torch.no_grad():
            features = model(img_tensor).tolist()

        all_features.append(features)
        all_positions.append(positions)

    return np.array(all_features), np.array(all_positions), xyxy


def _process_caption_data(caption_file, image_dir, max_length):
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

def _build_vocab(annotations, threshold=1):
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


def _build_caption_vector(annotations, word_index, max_length=24):
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


def _build_file_names(annotations):
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


def _build_image_indices(annotations, id_index):
    image_indices = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_indices[i] = id_index[image_id]

    return image_indices


if __name__ == "__main__":
    # maximum length of caption(number of word). 
    # if caption is longer than max_length, deleted.
    max_length = MAX_LENGTH
    # if word occurs less than word_count_threshold in training dataset, 
    # the word index is special unknown token.
    word_count_threshold = WORD_COUNT_THRESHOLD

    caption_path = DATA_PATH

    # about 110000 images and 5500000 captions for train dataset
    train_dataset = _process_caption_data(
        caption_file='../raw_data/MSCOCO/annotations/captions_train2017.json',
        image_dir='../raw_data/MSCOCO/image/train2017/',
        max_length=max_length)

    # about 5000 images and 25000 captions
    valid_dataset = _process_caption_data(
        caption_file='../raw_data/MSCOCO/annotations/captions_val2017.json',
        image_dir='../raw_data/MSCOCO/image/val2017/',
        max_length=max_length)

    # about 2500 images and 2500 captions for val / test dataset
    valid_cutoff = int(0.5 * len(valid_dataset))
    test_cutoff = int(len(valid_dataset))
    print('Finished processing caption data')

    save_pickle(train_dataset, f'{caption_path}/train/train.annotations.pkl')
    save_pickle(valid_dataset[:valid_cutoff], f'{caption_path}/valid/valid.annotations.pkl')
    save_pickle(valid_dataset[valid_cutoff:test_cutoff].reset_index(drop=True),
            f'{caption_path}/test/test.annotations.pkl')

    model = ResnetExtractor()
    model.eval()
    image_size = 224
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    tfms = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(norm_mean, norm_std),])

    for split in ['train', 'valid', 'test']:
        annotations = load_pickle(f'{caption_path}/{split}/{split}.annotations.pkl')

        if split == 'train':
            word_index = _build_vocab(annotations=annotations,
                                      threshold=word_count_threshold)
            save_pickle(word_index, f'{caption_path}/train/word_index.pkl')

        captions = _build_caption_vector(annotations=annotations,
                                         word_index=word_index,
                                         max_length=max_length)
        save_pickle(captions, f'{caption_path}/{split}/{split}.captions.pkl')

        file_names, id_index = _build_file_names(annotations=annotations)
        save_pickle(file_names, f'{caption_path}/{split}/{split}.file.names.pkl')

        image_indices = _build_image_indices(annotations=annotations,
                                             id_index=id_index)
        save_pickle(image_indices, f'{caption_path}/{split}/{split}.image.indices.pkl')

        # prepare reference captions to compute bleu scores later
        image_ids = {}
        feature_to_captions = {}

        i = -1
        for caption, image_id in zip(annotations['caption'], annotations['image_id']):
            if not image_id in image_ids:
                image_ids[image_id] = 0

                i += 1
                feature_to_captions[i] = []

            feature_to_captions[i].append(caption.lower() + ' .')

        save_pickle(feature_to_captions, f'{caption_path}/{split}/{split}.references.pkl')
        print(f"Finished building {split} caption dataset")

        # image features
        print(f"Building {split} image features...")
        image_path = list(annotations['file_name'].unique())
        n_examples = len(image_path)

        # i = 0
        # sp = 10000
        all_feats = np.ndarray([n_examples, NUM_OBJECT+1, 2048], dtype=np.float32)
        all_posit = np.ndarray([n_examples, NUM_OBJECT+1, 84], dtype=np.float32)

        feats_save_path = f'{caption_path}/{split}/{split}.features.hkl'
        posit_save_path = f'{caption_path}/{split}/{split}.positions.hkl'
        print(f"{feats_save_path}")
        
        for start, path in enumerate(tqdm(image_path)):
            features, positions, _ = image_feature(image_path=path,
                                                   model=model,
                                                   transforms=tfms,
                                                   image_size=image_size)
            end = start + 1
            all_feats[start:end, :] = features
            all_posit[start:end, :] = positions

        # use hickle to save huge feature vectors
        hickle.dump(all_feats, feats_save_path)
        hickle.dump(all_posit, posit_save_path)
        # print(f"Saved {save_path}..")
