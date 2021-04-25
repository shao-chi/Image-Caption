import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.config import PAD_IDX, WORD_TO_IDX_PATH
from core.metrics.cider.cider import Cider
from core.metrics.ciderD.ciderD import CiderD
from core.metrics.bleu.bleu import Bleu
from core.utils import decode_captions


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean', ignore_index=0):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight # weight parameter will act as the alpha parameter to balance class weights
        self.ignore_index = ignore_index

    def forward(self, output, target):
        ce_loss = F.cross_entropy(output, target,
                                  reduction=self.reduction,
                                  weight=self.weight,
                                  ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        return focal_loss


class ReinforcementLearningLoss(nn.Module):
    def __init__(self):
        super(ReinforcementLearningLoss, self).__init__()

        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.reward_criterion = RewardCriterion()
        self.structure_criterion = StructureCriterion()
        
        self.structure_loss_weight = 0.5

    def forward(self, model_output, sample_sequence, sample_logprobs, target):
        target = target[:, 1:].clone().long().contiguous()
        out = {}

        if self.structure_loss_weight < 1:
            model_output = model_output.view(-1, model_output.size(2))
            target_ = target.view(-1)

            language_model_loss = self.criterion(model_output, target_)
        else:
            language_model_loss = torch.FloatTensor(0)

        if self.structure_loss_weight > 0:
            structure_loss = self.structure_criterion(sample_logprobs, sample_sequence, target)
        else:
            structure_loss = {'loss': torch.FloatTensor(0),
                              'reward': torch.FloatTensor(0)}

        out['loss'] = (1 - self.structure_loss_weight) * language_model_loss + \
                        self.structure_loss_weight * structure_loss['loss']
        out['language_model_loss'] = language_model_loss
        out['structure_loss'] = structure_loss['loss']
        out['reward'] = structure_loss['reward']
        
        return out
        

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, output, sequence, reward):
        output = output.gather(2, sequence.unsqueeze(2)).squeeze(2)
        
        output = output.reshape(-1)
        reward = reward.reshape(-1)
        mask = (sequence > 0).to(output)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        loss = - output * reward * mask
        loss = torch.sum(loss) / torch.sum(mask)

        return loss


class StructureCriterion(nn.Module):
    """
    This loss is inspired by Classical Structured Prediction Losses for 
    Sequence to Sequence Learning (Edunov et al., 2018).
    """
    def __init__(self):
        super(StructureCriterion, self).__init__()

        word_to_idx = pickle.load(open(WORD_TO_IDX_PATH, 'rb'))
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        self.cider_reward_weight = 0.5
        self.ciderD_scorer = CiderD(df='coco-val')
        self.cider_scorer = Cider(df='coco-val')

        self.bleu_reward_weight = 0.5
        self.bleu_scorer = Bleu(4, print_=False)

        self.entropy_reward_weight = 0.5
        self.self_cider_reward_weight = 0.5

    def forward(self, output, sequence, target):
        out = {}

        mask = (sequence > 0).to(output)
        mask = torch.cat([mask.new_full((mask.size(0), 1), 1), mask[:, :-1]], 1)

        scores = self.get_scores(target, sequence)
        scores = torch.from_numpy(scores).type_as(output).view(-1, 1)
        out['reward'] = scores

        if self.entropy_reward_weight > 0:
            entropy = - (F.softmax(output, dim=2) * F.log_softmax(output, dim=2)).sum(2).data
            entropy = (entropy * mask).sum(1) / mask.sum(1)
            # print('entropy', entropy.mean().item())
            scores = scores + self.entropy_reward_weight * entropy.view(-1, 1)

        # Gather input: BxTxD -> BxT
        output = output.gather(2, sequence.unsqueeze(2)).squeeze(2)

        baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1])
        scores = scores - baseline

        # self cider used as reward to promote diversity (not working that much in this way)
        if self.self_cider_reward_weight > 0:
            _scores = self.get_self_cider_scores(target, sequence)
            _scores = torch.from_numpy(_scores).type_as(scores).view(-1, 1)
            _scores = _scores.expand_as(scores - 1)
            scores += self.self_cider_reward_weight * _scores
            
        output = - output * mask * scores.view(-1, 1)
        output = torch.sum(output) / torch.sum(mask)
    
        out['loss'] = output

        return out

    def get_scores(self, target, sequence):
        batch_size = sequence.size(0) # batch_size = sample_size * seq_per_img
        # seq_per_img = batch_size // len(target)

        sequence = sequence.data.cpu().numpy()
        res = self.decode_captions(sequence)
        gts = self.decode_captions(target.numpy())

        # res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
        res = {i: [res[i]] for i in range(batch_size)}
        gts = {i: [gts[i]] for i in range(batch_size)}

        if self.cider_reward_weight > 0:
            _, cider_scores = self.ciderD_scorer.compute_score(gts, res)
            # print('Cider scores:', _)
        else:
            cider_scores = 0

        if self.bleu_reward_weight > 0:
            try:
                _, bleu_scores = self.bleu_scorer.compute_score(gts, res)
                bleu_scores = np.array(bleu_scores[3])
                # print('Bleu scores:', _[3])
            except:
                bleu_scores = 0
        else:
            bleu_scores = 0

        scores = self.cider_reward_weight * cider_scores + self.bleu_reward_weight * bleu_scores

        return scores

    def get_self_cider_scores(self, target, sequence):
        # batch_size = sequence.size(0) # batch_size = sample_size * seq_per_img
        # seq_per_img = batch_size // len(data_gts)

        sequence = sequence.data.cpu().numpy()
        res = self.decode_captions(sequence)
        
        scores = []
        for i in range(len(target)):
            tmp = self.cider_scorer.my_self_cider([res[i:i+1]])

            def get_div(eigvals):
                eigvals = np.clip(eigvals, 0, None)
                sqrt = np.sqrt(eigvals).sum()
                log = np.log(len(eigvals))

                if sqrt == 0:
                    sqrt = 1e-8
                if log == 0:
                    log = 1e-8
                
                return -np.log(np.sqrt(eigvals[-1]) / (sqrt)) / log
            
            scores.append(get_div(np.linalg.eigvalsh(tmp[0]/10)))

        scores = np.array(scores)

        return scores

    def decode_captions(self, caption_vector):
        return decode_captions(captions=caption_vector,
                               index_to_word=self.idx_to_word)