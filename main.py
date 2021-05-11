import os
import time

import cv2
import fire
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from core.models import DEVICE, TRANSFORMER, SelfCriticNetwork
from core.config import *
from core.dataset import TrainDataset, TestDataset
from core.utils import save_pickle, write_scores
from core.evaluations import evaluate
from core.logger import TensorBoard_Writer

if CAPTION_MODEL == 'Transformer':
    MODEL = TRANSFORMER()
elif CAPTION_MODEL == 'RL_Transformer':
    MODEL = SelfCriticNetwork()


def train():
    writer = TensorBoard_Writer(LOG_PATH)

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

    for (t_features, t_positions, t_captions, _), \
        (v_features, v_positions, v_captions, _) \
            in zip(train_dataloader, valid_dataloader):
        eval_t_features = t_features
        eval_t_positions = t_positions
        eval_t_captions = t_captions
        
        eval_v_features = v_features
        eval_v_positions = v_positions
        eval_v_captions = v_captions
        break

    n_iter = len(train_dataloader)
    for epoch in range(1, NUM_EPOCH+1):
        print(f'Epoch {epoch}')

        for i, (batch_features,
                batch_positions,
                batch_captions,
                batch_image_idxs) in enumerate(tqdm(train_dataloader)):
            MODEL.train_step(batch_features=batch_features,
                             batch_positions=batch_positions,
                             batch_captions=batch_captions)
            
            if (i+1) % 100 == 0:
                train_loss = MODEL.compute_loss(object_features=eval_t_features,
                                                position_features=eval_t_positions,
                                                target_caption=eval_t_captions)
                valid_loss = MODEL.compute_loss(object_features=eval_v_features,
                                                position_features=eval_v_positions,
                                                target_caption=eval_v_captions)

                logs = {}
                for key in WRITE_LOG:
                    logs[key] = {'train': train_loss[key].cpu().mean().item(),
                                 'valid': valid_loss[key].cpu().mean().item()}

                writer.write_batch(logs=logs, step=i+n_iter*(epoch-1))
            
            if (i+1) % 2500 == 0:
                sample_caption, _ = MODEL.generate_caption(
                                        object_features=batch_features[:1],
                                        position_features=batch_positions[:1])

                print('\nSample caption: ', sample_caption[0])

                ground_truths = \
                    train_dataset.data['captions'] \
                        [train_dataset.data['image_idxs'] == batch_image_idxs[0].item()]
                ground_truths = MODEL.decode_captions(ground_truths)
                truths = ''
                for j, truth in enumerate(ground_truths):
                    truths += (truth + '\n')
                    print(f'Ground truth {j+1}: {truth}')
                
                writer.write_text(output=sample_caption[0],
                                  truths=truths,
                                  step=i+n_iter*(epoch-1))

        # evaluation
        valid_caption = [''] * valid_dataset.len_image
        logs = {key: {'train': 0, 'valid': 0} for key in WRITE_LOG}

        for i, ((batch_t_features, batch_t_positions, batch_t_captions, _), \
                (batch_v_features, batch_v_positions, batch_v_captions, batch_v_image_idxs)) \
                    in enumerate(zip(train_dataloader, valid_dataloader)):

            train_loss = MODEL.compute_loss(object_features=batch_t_features,
                                            position_features=batch_t_positions,
                                            target_caption=batch_t_captions)
                
            valid_loss = MODEL.compute_loss(object_features=batch_v_features,
                                            position_features=batch_v_positions,
                                            target_caption=batch_v_captions)
            
            for key in WRITE_LOG:
                logs[key]['train'] += train_loss[key].mean().item()
                logs[key]['valid'] += valid_loss[key].mean().item()

            captions, _ = MODEL.generate_caption(object_features=batch_v_features,
                                                 position_features=batch_v_positions)

            for idx, caption in zip(batch_v_image_idxs, captions):
                valid_caption[idx] = caption

        for key in WRITE_LOG:
            logs[key]['train'] /= len(valid_dataloader)
            logs[key]['valid'] /= len(valid_dataloader)

        save_pickle(valid_caption,
                    str(os.path.join(target_dir, "valid.candidate.captions.pkl")))
        scores = evaluate(target_dir=target_dir,
                          data_path=DATA_PATH,
                          split='valid',
                          get_scores=True)

        for key in WRITE_LOG:
            scores[key] = logs[key]

        print(f"\nTraining LOSS: {scores['loss']['train']}")
        print(f"Validation LOSS: {scores['loss']['valid']}")

        write_scores(scores=scores, path=OUTPUT_PATH, epoch=epoch, split='valid')

        writer.write_epoch(logs=scores, epoch=epoch)

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


def demo(image_path, beam_size=None, epoch=90, save_img=False, max_obj=False):
    start = time.time()
    features, positions, xyxy = MODEL.preprocess(image_path=image_path,
                                                 save_img=save_img,
                                                 max_obj=max_obj)
    
    feature = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
    position = torch.FloatTensor(positions).unsqueeze(0).to(DEVICE)

    model_path = os.path.join(OUTPUT_PATH, f'model/model_{epoch}.pt')

    MODEL.load(path=model_path)

    caption, attention_list = MODEL.generate_caption(object_features=feature,
                                                     position_features=position,
                                                     beam_size=beam_size)
    caption = caption[0]
    caption_length = len(caption.split(' '))

    if save_img and isinstance(attention_list, list):
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
                           int(obj_xyxy[0]):int(obj_xyxy[2])].copy()
                mask = mask * obj_attend + 255 * (1 - obj_attend)
                
                mask_list.append((mask, obj_attend, obj_xyxy))

            img = img * 0.2 + 255 * 0.8
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