import pickle

import cv2
import torch

from core.TRANSFORMER.model import Transformer
from core.config import *
from core.utils import decode_captions
from core.preprocess import image_feature_YOLOv5, image_feature_FasterRCNN, \
                            ResnetExtractor, FasterRCNNExtractor


if torch.cuda.is_available():
    device_name = "cuda:1"
else:
    device_name = "cpu"

DEVICE = torch.device(device_name)
print(f'Using {device_name}\n')


class MODEL_init:

    def __init__(self):
        super(MODEL_init, self).__init__()

    def train_step(self):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError

    def generate_caption(self):
        raise NotImplementedError

    def decode_captions(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def preprocess(self, image_path, save_img=False):
        if IMAGE_MODEL == 'YOLOv5':
            return image_feature_YOLOv5(image_path=image_path,
                                        save_img=save_img)
        
        elif IMAGE_MODEL == 'FasterRCNN':
            return image_feature_FasterRCNN(image_path=image_path,
                                            save_img=save_img)


class TRANSFORMER(MODEL_init):
    
    def __init__(self):
        super(TRANSFORMER, self).__init__()

        word_to_idx = pickle.load(open(WORD_TO_IDX_PATH, 'rb'))
        NUM_VOCAB = len(word_to_idx)
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        self.model = Transformer(num_vocab=NUM_VOCAB,
                                 max_length=MAX_LENGTH+2,
                                 encode_dim_positions=ENCODE_DIM_POSITIONS,
                                 encode_dim_features=ENCODE_DIM_FEATURES,
                                 encode_input_size=ENCODE_INPUT_SIZE,
                                 encode_q_k_dim=ENCODE_Q_K_DIM,
                                 encode_v_dim=ENCODE_V_DIM,
                                 encode_hidden_size=ENCODE_HIDDEN_SIZE,
                                 encode_num_blocks=ENCODE_NUM_BLOCKS,
                                 encode_num_heads=ENCODE_NUM_HEADS,
                                 dim_word_embedding=DIM_WORD_EMBEDDING,
                                 decode_input_size=DECODE_INPUT_SIZE,
                                 decode_q_k_dim=DECODE_Q_K_DIM,
                                 decode_v_dim=DECODE_V_DIM,
                                 decode_hidden_size=DECODE_HIDDEN_SIZE,
                                 decode_num_blocks=DECODE_NUM_BLOCKS,
                                 decode_num_heads=DECODE_NUM_HEADS,
                                 dropout=DROPOUT,
                                 device=DEVICE).to(DEVICE)
        self.optimizer = torch.optim.Adam((p for p in self.model.parameters() \
                                                if p.requires_grad),
                                          lr=LEARNING_RATE)

    def train_step(self, batch_features,
                         batch_positions,
                         batch_captions):
        self.optimizer.zero_grad()

        loss = self.model(object_features=batch_features.to(DEVICE),
                          position_features=batch_positions.to(DEVICE),
                          target_caption=batch_captions.to(DEVICE))

        loss.backward()
        self.optimizer.step()

    def compute_loss(self, object_features,
                           position_features,
                           target_caption):
        with torch.no_grad():
            return \
                self.model(object_features=object_features.to(DEVICE),
                           position_features=position_features.to(DEVICE),
                           target_caption=target_caption.to(DEVICE)).cpu().item()

    def generate_caption(self, object_features,
                               position_features,
                               beam_size=None):
        if beam_size in [None, 1]:
            caption_vector, attention_list = \
                            self.model.generate_caption_vector(
                                object_features=object_features.to(DEVICE),
                                position_features=position_features.to(DEVICE))

            return self.decode_captions(caption_vector.cpu().numpy()), \
                    attention_list

        elif isinstance(beam_size, int) and beam_size > 1:
            caption_vector = self.model.beam_search(
                                object_features=object_features.to(DEVICE),
                                position_features=position_features.to(DEVICE),
                                beam_size=beam_size)
        
            return self.decode_captions(caption_vector.cpu().numpy()), None

        else:
            assert isinstance(beam_size, int)
            assert beam_size > 1 or beam_size in [None, 1]

    def decode_captions(self, caption_vector):
        return decode_captions(captions=caption_vector,
                               index_to_word=self.idx_to_word)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()