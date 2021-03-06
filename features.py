import os

import cv2
import hickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.preprocess import *
from core.utils import save_pickle, load_pickle
from core.config import DATA_PATH, IMAGE_MODEL, ENCODE_DIM_FEATURES, \
                        ENCODE_DIM_POSITIONS, NUM_OBJECT, MAX_LENGTH, \
                        WORD_COUNT_THRESHOLD
from core.dataset import ImagePreprocessDataset

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(DATA_PATH, split))

    if not os.path.exists('./data/train.annotations.pkl'):
        print("./data/train.annotations.pkl doesn't exist.")
        # about 110000 images and 5500000 captions for train dataset
        train_dataset = process_caption_data(
            caption_file='./raw_data/MSCOCO/annotations/captions_train2017.json',
            image_dir='./raw_data/MSCOCO/image/train2017/',
            max_length=MAX_LENGTH)
        save_pickle(train_dataset, './data/train.annotations.pkl')
        print('Finished processing caption data for training.')

    if not os.path.exists('./data/test.annotations.pkl') or \
            not os.path.exists('./data/valid.annotations.pkl'):
        print("./data/valid or test.annotations.pkl doesn't exist.")
        # about 5000 images and 25000 captions
        valid_dataset = process_caption_data(
            caption_file='./raw_data/MSCOCO/annotations/captions_val2017.json',
            image_dir='./raw_data/MSCOCO/image/val2017/',
            max_length=MAX_LENGTH)

        # about 2500 images and 2500 captions for val / test dataset
        valid_cutoff = int(0.5 * len(valid_dataset))
        test_cutoff = int(len(valid_dataset))
        print('Finished processing caption data for validation and testing')

        save_pickle(valid_dataset[:valid_cutoff], './data/valid.annotations.pkl')
        save_pickle(valid_dataset[valid_cutoff:test_cutoff].reset_index(drop=True),
                './data/test.annotations.pkl')

    for split in ['train', 'valid', 'test']:
        annotations = load_pickle(f'./data/{split}.annotations.pkl')

        if split == 'train':
            word_index = build_vocab(annotations=annotations,
                                      threshold=WORD_COUNT_THRESHOLD)
            save_pickle(word_index, f'{DATA_PATH}/train/word_index.pkl')

        captions = build_caption_vector(annotations=annotations,
                                         word_index=word_index,
                                         max_length=MAX_LENGTH)
        save_pickle(captions, f'{DATA_PATH}/{split}/{split}.captions.pkl')

        file_names, id_index = build_file_names(annotations=annotations)
        save_pickle(file_names, f'{DATA_PATH}/{split}/{split}.file.names.pkl')

        image_indices = build_image_indices(annotations=annotations,
                                             id_index=id_index)
        save_pickle(image_indices, f'{DATA_PATH}/{split}/{split}.image.indices.pkl')

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

        save_pickle(feature_to_captions, f'{DATA_PATH}/{split}/{split}.references.pkl')
        print(f"Finished building {split} caption dataset")

        # image features
        print(f"Building {split} image features...")
        image_path = list(annotations['file_name'].unique())
        n_examples = len(image_path)

        batch_size = 100
        image_dataset = ImagePreprocessDataset(path_list=image_path,
                                               model=IMAGE_MODEL)
        image_dataloader = DataLoader(image_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=1)

        # i = 0
        # sp = 10000
        all_feats = np.ndarray([n_examples, NUM_OBJECT+1, ENCODE_DIM_FEATURES],
                               dtype=np.float32)
        all_posit = np.ndarray([n_examples, NUM_OBJECT+1, ENCODE_DIM_POSITIONS],
                               dtype=np.float32)

        feats_save_path = f'{DATA_PATH}/{split}/{split}.features.hkl'
        posit_save_path = f'{DATA_PATH}/{split}/{split}.positions.hkl'
        print(f"Start saving {feats_save_path}")
        
        for start, (features, positions) in enumerate(tqdm(image_dataloader)):
            start = start * batch_size
            end = start + len(positions)
            all_feats[start:end, :] = features
            all_posit[start:end, :] = positions

        # use hickle to save huge feature vectors
        hickle.dump(all_feats, feats_save_path)
        hickle.dump(all_posit, posit_save_path)
        print(f"Saved {feats_save_path} and {posit_save_path}..")
