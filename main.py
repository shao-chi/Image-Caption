import os
import time

import cv2
import fire
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from core.models import DEVICE, TRANSFORMER, SelfCriticNetwork
from core.config import *
from core.dataset import TrainDataset, TestDataset
from core.utils import save_pickle, write_scores
from core.evaluations import evaluate

if CAPTION_MODEL == 'Transformer':
    MODEL = TRANSFORMER()
elif CAPTION_MODEL == 'RL_Transformer':
    MODEL = SelfCriticNetwork()


def train():
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
            MODEL.train_step(batch_features=batch_features,
                             batch_positions=batch_positions,
                             batch_captions=batch_captions)
            
            if i % 20 == 0:
                if CAPTION_MODEL == 'Transformer':
                    t_loss = MODEL.compute_loss(object_features=eval_t_features,
                                                position_features=eval_t_positions,
                                                target_caption=eval_t_captions)
                    v_loss = MODEL.compute_loss(object_features=eval_v_features,
                                                position_features=eval_v_positions,
                                                target_caption=eval_v_captions)
                    writer.add_scalars('LOSS/BATCH',
                                       {'Train': t_loss, 'Valid': v_loss},
                                       i+n_iter*(epoch-1))

                elif CAPTION_MODEL == 'RL_Transformer':
                    t_r_loss, t_reward, t_c_loss = MODEL.compute_loss(object_features=eval_t_features,
                                                                      position_features=eval_t_positions,
                                                                      target_caption=eval_t_captions)
                    v_r_loss, v_reward, v_c_loss =  MODEL.compute_loss(object_features=eval_v_features,
                                                                       position_features=eval_v_positions,
                                                                       target_caption=eval_v_captions)
                    writer.add_scalars('REWARD/BATCH',
                                        {'Train': t_reward, 'Valid': v_reward},
                                        i+n_iter*(epoch-1))
                    writer.add_scalars('LOSS/BATCH_REWARD',
                                       {'Train': t_r_loss, 'Valid': v_r_loss},
                                       i+n_iter*(epoch-1))
                    writer.add_scalars('LOSS/BATCH_CAPTION',
                                       {'Train': t_c_loss, 'Valid': v_c_loss},
                                       i+n_iter*(epoch-1))
            
            if (i+1) % 2500 == 0:
                sample_caption, _ = MODEL.generate_caption(
                                    object_features=batch_features[:1],
                                    position_features=batch_positions[:1])

                writer.add_text('CAPTION/Sample_caption',
                                sample_caption[0],
                                i+n_iter*(epoch-1))
                print('\nSample caption: ', sample_caption[0])

                ground_truths = \
                    train_dataset.data['captions'] \
                        [train_dataset.data['image_idxs'] == batch_image_idxs[0].item()]
                ground_truths = MODEL.decode_captions(ground_truths)
                truths = ''
                for j, truth in enumerate(ground_truths):
                    truths += (truth + '\n')
                    print(f'Ground truth {j+1}: {truth}')
                
                writer.add_text('CAPTION/Ground_truth',
                                truths,
                                i+n_iter*(epoch-1))

        # evaluation
        if CAPTION_MODEL == 'Transformer':
            train_loss = 0
        elif CAPTION_MODEL == 'RL_Transformer':
            train_r_loss = 0
            train_c_loss = 0
            train_reward = 0

        for i, (batch_features,
                batch_positions,
                batch_captions, _) in enumerate(train_dataloader):

            if i < len(valid_dataloader):
                if CAPTION_MODEL == 'Transformer':
                    train_loss += MODEL.compute_loss(object_features=batch_features,
                                                     position_features=batch_positions,
                                                     target_caption=batch_captions)
                
                elif CAPTION_MODEL == 'RL_Transformer':
                    r_loss, reward, c_loss = MODEL.compute_loss(object_features=batch_features,
                                                                position_features=batch_positions,
                                                                target_caption=batch_captions)
                    train_r_loss += r_loss
                    train_reward += reward
                    train_c_loss += c_loss
                
            else:
                break

        if CAPTION_MODEL == 'Transformer':
            valid_loss = 0
        elif CAPTION_MODEL == 'RL_Transformer':
            valid_r_loss = 0
            valid_c_loss = 0
            valid_reward = 0

        valid_caption = [''] * valid_dataset.len_image
        for batch_features, \
            batch_positions, \
            batch_captions, \
            batch_image_idxs in valid_dataloader:
            if CAPTION_MODEL == 'Transformer':
                valid_loss += MODEL.compute_loss(object_features=batch_features,
                                                 position_features=batch_positions,
                                                 target_caption=batch_captions)
                
            elif CAPTION_MODEL == 'RL_Transformer':
                r_loss, reward, c_loss = MODEL.compute_loss(object_features=batch_features,
                                                            position_features=batch_positions,
                                                            target_caption=batch_captions)
                valid_r_loss += r_loss
                valid_reward += reward
                valid_c_loss += c_loss

            captions, _ = MODEL.generate_caption(object_features=batch_features,
                                                 position_features=batch_positions)

            for i, (idx, caption) in enumerate(zip(batch_image_idxs, captions)):
                valid_caption[idx] = caption

        save_pickle(valid_caption,
                    str(os.path.join(target_dir, "valid.candidate.captions.pkl")))
        scores = evaluate(target_dir=target_dir,
                          data_path=DATA_PATH,
                          split='valid',
                          get_scores=True)

        if CAPTION_MODEL == 'Transformer':
            train_loss = train_loss / len(valid_dataloader)
            valid_loss = valid_loss / len(valid_dataloader)
            print(f"\nTraining LOSS: {train_loss}")
            print(f"Validation LOSS: {valid_loss}")

            scores['train_loss'] = train_loss
            scores['valid_loss'] = valid_loss
            writer.add_scalars('LOSS/EPOCH', {'Train': scores['train_loss'],
                                              'Valid': scores['valid_loss']}, epoch)

        elif CAPTION_MODEL == 'RL_Transformer':
            train_r_loss = train_r_loss / len(valid_dataloader)
            valid_r_loss = valid_r_loss / len(valid_dataloader)
            print(f"\nTraining LOSS (REWARD): {train_r_loss}")
            print(f"Validation LOSS (REWARD): {valid_r_loss}")
            train_c_loss = train_c_loss / len(valid_dataloader)
            valid_c_loss = valid_c_loss / len(valid_dataloader)
            print(f"Training LOSS (CAPTION): {train_c_loss}")
            print(f"Validation LOSS (CAPTION): {valid_c_loss}")
            train_reward = train_reward / len(valid_dataloader)
            valid_reward = valid_reward / len(valid_dataloader)
            print(f"Training REWARD: {train_reward}")
            print(f"Validation REWARD\n: {valid_reward}")

            scores['train_reward_loss'] = train_r_loss
            scores['valid_reward_loss'] = valid_r_loss
            writer.add_scalars('LOSS/EPOCH_REWARD', {'Train': train_r_loss,
                                                     'Valid': valid_r_loss}, epoch)
            scores['train_caption_loss'] = train_c_loss
            scores['valid_caption_loss'] = valid_c_loss
            writer.add_scalars('LOSS/EPOCH', {'Train': train_c_loss,
                                              'Valid': valid_c_loss}, epoch)
            scores['train_reward'] = train_reward
            scores['valid_reward'] = valid_reward
            writer.add_scalars('REWARD/EPOCH', {'Train': train_reward,
                                                'Valid': valid_reward}, epoch)

        write_scores(scores=scores, path=OUTPUT_PATH, epoch=epoch, split='valid')

        for score_name, score in scores.items():
            if score_name not in ['train_loss', 'valid_loss',
                                  'train_reward', 'valid_reward',
                                  'train_reward_loss', 'valid_reward_loss',
                                  'train_caption_loss', 'valid_caption_loss']:
                writer.add_scalar(f'EVALUATION/{score_name}', score, epoch)

        MODEL.save(path=os.path.join(model_dir, f'model_{epoch}.pt'))

    writer.close()


def evaluation(split='test', epoch=90, beam_size=None):
    model_path = os.path.join(OUTPUT_PATH, f'model/model_{epoch}.pt')

    MODEL.load(path=model_path)

    test_dataset = TestDataset(data_path=DATA_PATH, split=split)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

    target_dir = os.path.join(DATA_PATH, f"{split}/{OUTPUT_NAME}/")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    test_caption = [''] * test_dataset.len_image

    for batch_features, \
        batch_positions, \
        batch_image_idxs in tqdm(test_dataloader):

        captions, _ = MODEL.generate_caption(object_features=batch_features,
                                             position_features=batch_positions,
                                             beam_size=beam_size)

        for i, idx in enumerate(batch_image_idxs):
            test_caption[idx] = captions[i]

    save_pickle(test_caption,
                str(os.path.join(target_dir, f"{split}.candidate.captions.pkl")))

    scores = evaluate(target_dir=target_dir,
                      data_path=DATA_PATH,
                      split=split,
                      get_scores=True)
    write_scores(scores=scores, path=OUTPUT_PATH, epoch=epoch, split=split)


def demo(image_path, beam_size=None, epoch=90, save_img=False):
    start = time.time()
    features, positions, xyxy = MODEL.preprocess(image_path=image_path,
                                                 save_img=save_img)
    
    feature = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
    position = torch.FloatTensor(positions).unsqueeze(0).to(DEVICE)

    model_path = os.path.join(OUTPUT_PATH, f'model/model_{epoch}.pt')

    MODEL.load(path=model_path)

    caption, attention_list = MODEL.generate_caption(object_features=feature,
                                                     position_features=position,
                                                     beam_size=beam_size)
    caption = caption[0]
    caption_length = len(caption.split(' '))

    if save_img and isinstance(attention_list, list) :
        attention_list = np.array(attention_list) \
                                .reshape(MAX_LENGTH+1, NUM_OBJECT+1)

        _, image_name = os.path.split(image_path)
        image_dir = image_name.split('.')[0]
        for i, attention in enumerate(attention_list):
            img = cv2.imread(image_path)

            mask_list = []
            for obj_attend, obj_xyxy in zip(attention[1:], xyxy):
                if obj_attend == 0:
                    continue

                c1 = (int(obj_xyxy[0]), int(obj_xyxy[1]))
                c2 = (int(obj_xyxy[2]), int(obj_xyxy[3]))

                mask = img[int(obj_xyxy[1]):int(obj_xyxy[3]),
                           int(obj_xyxy[0]):int(obj_xyxy[2])]

                # zeros = np.zeros((img.shape), dtype=np.uint8)
                zeros_mask = cv2.rectangle(mask,
                                           pt1=(0, 0),
                                           pt2=(mask.shape[1]-1, mask.shape[0]-1),
                                           color=(255, 255, 255),
                                           thickness=-1)
                mask = cv2.addWeighted(mask, obj_attend, zeros_mask, 1-obj_attend, gamma=0)
                cv2.imwrite(f'./demo/{image_dir}/{IMAGE_MODEL}/{len(mask_list)+1}_mask.jpg', mask)
                mask_list.append((mask, obj_attend, obj_xyxy))

            mask_list = sorted(mask_list, key=lambda k: k[1])
            for mask, _, obj_xyxy in mask_list:
                img[int(obj_xyxy[1]):int(obj_xyxy[3]),
                    int(obj_xyxy[0]):int(obj_xyxy[2])] = mask

            cv2.imwrite(f'./demo/{image_dir}/{IMAGE_MODEL}/{i+1}_{image_name}', img)

            if i == (caption_length - 1):
                break

    print("Generated Caption:", caption)
    print("Spending Time:", time.time() - start)


if __name__ == '__main__':
    fire.Fire()