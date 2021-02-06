import pickle
import os

from .metrics.bleu.bleu import Bleu
from .metrics.rouge.rouge import Rouge
from .metrics.cider.cider import Cider
from .metrics.meteor.meteor import Meteor
from .metrics.spice.spice import Spice


def _score(ref_captions, hypo_captions):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Spice(), "SPICE")
    ]

    final_scores = dict()
    for scorer, method in scorers:
        scores, _ = scorer.compute_score(gts=ref_captions,
                                         res=hypo_captions)

        if isinstance(scores, list):
            for m, s in zip(method, scores):
                final_scores[m] = s

        else:
            final_scores[method] = scores

    return final_scores


def evaluate(target_dir, data_path, split='valid', get_scores=False):
    reference_path = os.path.join(data_path, f"{split}/{split}.references.pkl")
    candidate_path = os.path.join(target_dir, f"{split}.candidate.captions.pkl")

    # load caption data
    with open(reference_path, 'rb') as file_:
        reference_captions = pickle.load(file_)

    with open(candidate_path, 'rb') as file_:
        candidate_captions = pickle.load(file_)

    # make dictionary
    hypo_captions = dict()
    for i, caption in enumerate(candidate_captions):
        hypo_captions[i] = [caption]

    # compute score
    final_scores = _score(ref_captions=reference_captions,
                          hypo_captions=hypo_captions)

    # print out scores
    print('\n')
    for score_name, score in final_scores.items():
        print(f"{score_name}:\t{score}")
    print('\n')

    if get_scores:
        return final_scores
