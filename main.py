import os
import pickle

import fire
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from core.model.transformer import Transformer
from core.settings import *
from core.dataset import TrainDataset, TestDataset
from core.utils import decode_captions, write_bleu, save_pickle
from core.evaluations import evaluate
from preprocess import image_feature, ResnetExtractor

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:1")
    print('\nUsing cuda:1\n')

else:
    DEVICE = torch.device("cpu")
    print('\nUsing cpu\n')


class MODEL:
    def __init__(self):
        super(MODEL, self).__init__()

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
        caption_vector = self.model.generate_caption_vector(
                                object_features=object_features.to(DEVICE),
                                position_features=position_features.to(DEVICE),
                                beam_size=beam_size).cpu()
        
        return self.decode_captions(caption_vector.numpy())

    def decode_captions(self, caption_vector):
        return decode_captions(captions=caption_vector,
                               index_to_word=self.idx_to_word)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        self.model.eval()


def train():
    model = MODEL()
    writer = SummaryWriter(LOG_PATH)

    model_dir = os.path.join(OUTPUT_PATH, 'model/')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    target_dir = os.path.join(DATA_PATH, f"valid/{OUTPUT_NAME}/")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    train_dataset = TrainDataset(data_path=DATA_PATH, split='train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    valid_dataset = TrainDataset(data_path=DATA_PATH, split='valid')
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)

    n_iter = len(train_dataloader)
    for features, positions, captions, _ in train_dataloader:
        eval_t_features = features
        eval_t_positions = positions
        eval_t_captions = captions
        break
    for features, positions, captions, _ in valid_dataloader:
        eval_v_features = features
        eval_v_positions = positions
        eval_v_captions = captions
        break

    for epoch in range(1, NUM_EPOCH+1):
        print(f'Epoch {epoch}')

        for i, (batch_features,
                batch_positions,
                batch_captions,
                batch_image_idxs) in enumerate(tqdm(train_dataloader)):

            model.train_step(batch_features=batch_features,
                             batch_positions=batch_positions,
                             batch_captions=batch_captions)
            
            if i % 20 == 0:
                t_loss = model.compute_loss(object_features=eval_t_features,
                                            position_features=eval_t_positions,
                                            target_caption=eval_t_captions)
                v_loss = model.compute_loss(object_features=eval_v_features,
                                            position_features=eval_v_positions,
                                            target_caption=eval_v_captions)
                writer.add_scalars('LOSS/BATCH',
                                   {'Train': t_loss, 'Valid': v_loss},
                                   i+n_iter*(epoch-1))
            
            if (i+1) % 2500 == 0:
                sample_caption = model.generate_caption(
                                    object_features=batch_features[:1],
                                    position_features=batch_positions[:1])

                writer.add_text('CAPTION/Sample_caption',
                                sample_caption[0], 
                                i+n_iter*(epoch-1))
                print('\nSample caption: ', sample_caption[0])

                ground_truths = \
                    train_dataset.data['captions'] \
                        [train_dataset.data['image_idxs'] == batch_image_idxs[0].item()]
                ground_truths = model.decode_captions(ground_truths)
                truths = ''
                for j, truth in enumerate(ground_truths):
                    truths += (truth + '\n')
                    print(f'Ground truth {j+1}: {truth}')
                
                writer.add_text('CAPTION/Ground_truth',
                                truths,
                                i+n_iter*(epoch-1))
            # break

        # evaluation
        train_loss = 0
        for i, (batch_features,
                batch_positions,
                batch_captions, _) in enumerate(train_dataloader):

            if i < len(valid_dataloader):
                train_loss += model.compute_loss(
                                object_features=batch_features,
                                position_features=batch_positions,
                                target_caption=batch_captions)
            else:
                break

        valid_loss = 0
        valid_caption = [''] * valid_dataset.len_image

        for batch_features, \
            batch_positions, \
            batch_captions, \
            batch_image_idxs in valid_dataloader:

            valid_loss += model.compute_loss(object_features=batch_features,
                                             position_features=batch_positions,
                                             target_caption=batch_captions)
            captions = model.generate_caption(object_features=batch_features,
                                              position_features=batch_positions)

            for i, idx in enumerate(batch_image_idxs):
                valid_caption[idx] = captions[i]

        save_pickle(valid_caption,
                    str(os.path.join(target_dir, "valid.candidate.captions.pkl")))

        train_loss = train_loss / len(valid_dataloader)
        valid_loss = valid_loss / len(valid_dataloader)
        print(f"\nTraining LOSS: {train_loss}")
        print(f"Validation LOSS: {valid_loss}\n")

        scores = evaluate(target_dir=target_dir,
                          data_path=DATA_PATH,
                          split='valid',
                          get_scores=True)
        scores['train_loss'] = train_loss
        scores['valid_loss'] = valid_loss
        write_bleu(scores=scores, path=OUTPUT_PATH, epoch=epoch)

        writer.add_scalars('LOSS/EPOCH', {'Train': scores['train_loss'],
                                          'Valid': scores['valid_loss']}, epoch)
        writer.add_scalar('EVALUATION/BLEU_1', scores['Bleu_1'], epoch)
        writer.add_scalar('EVALUATION/BLEU_2', scores['Bleu_2'], epoch)
        writer.add_scalar('EVALUATION/BLEU_3', scores['Bleu_3'], epoch)
        writer.add_scalar('EVALUATION/BLEU_4', scores['Bleu_4'], epoch)
        writer.add_scalar('EVALUATION/METEOR', scores['METEOR'], epoch)
        writer.add_scalar('EVALUATION/ROUGE_L', scores['ROUGE_L'], epoch)
        writer.add_scalar('EVALUATION/CIDEr', scores['CIDEr'], epoch)

        model.save(path=os.path.join(model_dir, f'model_{epoch}.pt'))

    writer.close()


def evaluation(split='test', epoch=90, beam_size=1):
    model_path = os.path.join(OUTPUT_PATH, f'model/model_{epoch}.pt')

    model = MODEL()
    model.load(path=model_path)

    test_dataset = TestDataset(data_path=DATA_PATH, split=split)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

    target_dir = os.path.join(DATA_PATH, f"{split}/{OUTPUT_NAME}/")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    test_caption = [''] * test_dataset.len_image

    for i, (batch_features,
            batch_positions,
            batch_image_idxs) in enumerate(tqdm(test_dataloader)):

        captions = model.generate_caption(object_features=batch_features,
                                          position_features=batch_positions,
                                          beam_size=beam_size)

        for i, idx in enumerate(batch_image_idxs):
            test_caption[idx] = captions[i]

    save_pickle(test_caption,
                str(os.path.join(target_dir, f"{split}.candidate.captions.pkl")))
    evaluate(target_dir=target_dir,
             data_path=DATA_PATH,
             split=split,
             get_scores=False)


def demo(image_path, beam_size, epoch=90):
    resnet_model = ResnetExtractor()
    resnet_model.eval()
    image_size = 224
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    tfms = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(norm_mean, norm_std),])

    features, positions = image_feature(image_path=image_path,
                                        model=resnet_model,
                                        transforms=tfms,
                                        image_size=image_size,
                                        save_img=True)
    feature = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
    position = torch.FloatTensor(positions).unsqueeze(0).to(DEVICE)


    model_path = os.path.join(OUTPUT_PATH, f'model/model_{epoch}.pt')

    model = MODEL()
    model.load(path=model_path)

    caption = model.generate_caption(object_features=feature,
                                     position_features=position,
                                     beam_size=beam_size)
    
    print("Generated Caption: ", caption[0])


if __name__ == '__main__':
    fire.Fire()