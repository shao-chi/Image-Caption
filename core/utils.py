import time
import os
import configparser

import numpy as np
import pickle
import hickle


def getConfig():
    config = configparser.ConfigParser()
    config.read('%s/core/config.cfg' % (os.getcwd()))
    
    return config


def load_pickle(path):
    with open(path, 'rb') as f:
        file_ = pickle.load(f)
        print(f'Loaded {path}..')

        return file_


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
        print(f'Saved {path}..')
        

def load_coco_data(data_path='./data/MSCOCO', split='train'):
    data_path = os.path.join(data_path, split)
    start_time = time.time()

    features_path = os.path.join(data_path, f'{split}.features.hkl')
    positions_path = os.path.join(data_path, f'{split}.positions.hkl')

    pickle_path = {}
    pickle_path['file_names'] = os.path.join(data_path, f'{split}.file.names.pkl')
    pickle_path['captions'] = os.path.join(data_path, f'{split}.captions.pkl')
    pickle_path['image_idxs'] = os.path.join(data_path, f'{split}.image.indices.pkl')

    data = {}
    data['features'] = hickle.load(features_path)
    data['positions'] = hickle.load(positions_path)

    for data_key, path in pickle_path.items():
        data[data_key] = load_pickle(path=path)

    if split == 'train':
        wordidx_path = os.path.join(data_path, 'word_index.pkl')
        data['word_to_idx'] = load_pickle(wordidx_path)
          
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(key, type(value), value.shape, value.dtype)
        else:
            print(key, type(value), len(value))

    end_time = time.time()
    print("Elapse time: %.2f" %(end_time - start_time))

    return data
    

def decode_captions(captions, index_to_word):
    if captions.ndim == 1:
        caption_length = captions.shape[0]
        caption_number = 1

    else:
        caption_number, caption_length = captions.shape

    decoded = []
    for i in range(caption_number):
        words = []

        for t in range(caption_length):
            if captions.ndim == 1:
                word = index_to_word[captions[t]]

            else:
                word = index_to_word[captions[i, t]]

            if word == '<START>' and t == 0:
                continue

            elif word == '<END>':
                words.append('.')
                break

            elif word != '<NULL>':
                words.append(word)

        decoded.append(' '.join(words))

    return decoded


# def sample_coco_minibatch(data, batch_size):
#     data_size = data['features'].shape[0]

#     if data_size > 1:
#         mask = np.random.choice(data_size, batch_size)
#     else:
#         mask = [0]

#     features = data['features'][mask]
#     file_names = data['file_names'][mask]

#     return features, file_names, mask


def write_scores(scores, path, epoch):
    if epoch == 1:
        file_mode = 'w'

    else:
        file_mode = 'a'

    with open(os.path.join(path, 'valid_scores.txt'), file_mode) as f:
        f.write(f'Epoch {epoch}\n')

        for score_name, score in scores.items():
            f.write(f"{score_name}: {score}\n")