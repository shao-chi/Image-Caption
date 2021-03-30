import pickle

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.TRANSFORMER.model import Transformer
from core.TRANSFORMER.model_RL import PolicyNetwork
from core.config import *
from core.utils import decode_captions
from core.preprocess import image_feature_YOLOv5, image_feature_FasterRCNN, \
                            ResnetExtractor, FasterRCNNExtractor
from core.metrics.cider.cider import Cider


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
                                 device=DEVICE,
                                 move_first_image_feature=MOVE_FIRST_IMAGE_FAETURE,
                                 split_position=SPLIT_POSITION).to(DEVICE)
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


class SelfCriticNetwork(MODEL_init):
    def __init__(self):
        super(SelfCriticNetwork, self).__init__()

        word_to_idx = pickle.load(open(WORD_TO_IDX_PATH, 'rb'))
        NUM_VOCAB = len(word_to_idx)
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        self.policy_net = PolicyNetwork(num_vocab=NUM_VOCAB,
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
                                        device=DEVICE,
                                        move_first_image_feature=MOVE_FIRST_IMAGE_FAETURE,
                                        split_position=SPLIT_POSITION).to(DEVICE)
        
        self.scorer = Cider()
        self.reward_loss = RewardCriterion()
        self.caption_loss = nn.NLLLoss()
        self.optimizer = torch.optim.Adam((p for p in self.policy_net.parameters() \
                                                if p.requires_grad),
                                          lr=LEARNING_RATE)
        
    def train_step(self, batch_features,
                         batch_positions,
                         batch_captions):
        self.optimizer.zero_grad()

        output, log_probs = self.policy_net(object_features=batch_features.to(DEVICE),
                                            position_features=batch_positions.to(DEVICE),
                                            target_caption=batch_captions.to(DEVICE),
                                            greedy=False)
        reward = self.get_rewards(object_features=batch_features.to(DEVICE),
                                  position_features=batch_positions.to(DEVICE),
                                  output_caption=output,
                                  target_caption=batch_captions.to(DEVICE))
                                  
        r_loss = self.reward_loss(log_probs,
                                  batch_captions.to(DEVICE),
                                  torch.from_numpy(reward).float().to(DEVICE).detach())
        batch_captions = batch_captions.clone().long().contiguous().view(-1)
        c_loss = self.caption_loss(log_probs.view(-1, log_probs.size(2)), batch_captions.to(DEVICE))

        r_loss.backward(retain_graph=True)
        c_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def get_rewards(self, object_features, position_features, output_caption, target_caption):
        batch_size = len(output_caption)

        greedy_output, _ = self.policy_net.sample(object_features=object_features,
                                                  position_features=position_features,
                                                  target_caption=target_caption,
                                                  greedy=True)
        hypo_captions = dict()
        for i, caption in enumerate(output_caption.detach().cpu().numpy()):
            hypo_captions[i] = self.decode_captions(caption)
        for i, caption in enumerate(greedy_output):
            hypo_captions[i+batch_size] = self.decode_captions(caption)

        reference_captions = dict()
        for i, caption in enumerate(target_caption.detach().cpu().numpy()):
            reference_captions[i] = self.decode_captions(caption)
        for i, caption in enumerate(target_caption.detach().cpu().numpy()):
            reference_captions[i+batch_size] = self.decode_captions(caption)

        _, scores = self.scorer.compute_score(reference_captions, hypo_captions)
        scores = scores[:batch_size] - scores[batch_size:]
        rewards = np.repeat(scores[:, np.newaxis], output_caption.shape[1], 1)

        return rewards

    def compute_loss(self, object_features,
                           position_features,
                           target_caption):
        with torch.no_grad():
            output, log_probs = \
                self.policy_net(object_features=object_features.to(DEVICE),
                                position_features=position_features.to(DEVICE),
                                target_caption=target_caption.to(DEVICE),
                                greedy=False)
            reward = self.get_rewards(object_features=object_features.to(DEVICE),
                                      position_features=position_features.to(DEVICE),
                                      output_caption=output,
                                      target_caption=target_caption.to(DEVICE))
            r_loss = self.reward_loss(log_probs,
                                    target_caption.to(DEVICE),
                                    torch.from_numpy(reward).float().to(DEVICE).detach()).item()

            target_caption = target_caption.clone().long().contiguous().view(-1).to(DEVICE)
            c_loss = self.caption_loss(log_probs.view(-1, log_probs.size(2)), target_caption)

            return r_loss, np.mean(reward[:, 0]), c_loss

    def generate_caption(self, object_features,
                               position_features,
                               beam_size=None):
        if beam_size in [None, 1]:
            caption_vector, attention_list = \
                            self.policy_net.generate_caption_vector(
                                object_features=object_features.to(DEVICE),
                                position_features=position_features.to(DEVICE))

            return self.decode_captions(caption_vector.cpu().numpy()), \
                    attention_list

        elif isinstance(beam_size, int) and beam_size > 1:
            caption_vector = self.policy_net.beam_search(
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
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.eval()

    def preprocess(self, image_path, save_img=False):
        if IMAGE_MODEL == 'YOLOv5':
            return image_feature_YOLOv5(image_path=image_path,
                                        save_img=save_img)
        
        elif IMAGE_MODEL == 'FasterRCNN':
            return image_feature_FasterRCNN(image_path=image_path,
                                            save_img=save_img)

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, log_probs, target_caption, reward):
        log_probs = log_probs.sum(2).contiguous().view(-1)
        reward = reward.contiguous().view(-1)

        mask = (target_caption > PAD_IDX).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).contiguous().view(-1)

        output = - log_probs * (1 + reward) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output