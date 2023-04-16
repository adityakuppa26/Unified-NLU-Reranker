import numpy as np
import pandas as pd
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

class AnnotatedDataset(Dataset):
    """ Annotated Dataset """

    def __init__(self, score_csv, idx_csv, transform = None) -> None:
        super().__init__()
        """
        Args:
            score_csv (string): Path to csv file with scores
            idx_csv (string): Path to csv file with indexes representing hypotheses
        """
        self.scores   = pd.read_csv(score_csv)
        self.idx      = pd.read_csv(idx_csv)
        print('---Annotated:', self.scores.shape, self.idx.shape)
        self.idx['ground_truth'] = self.idx['ground_truth'].apply(lambda x: ast.literal_eval(x))
        self.labels   = self.idx['gt_idx']

        self.Hall     = int(self.scores.shape[-1]/3)
        self.transform = transform
        # print('Hall:', self.Hall)

    def __len__(self)->int:
        return len(self.scores)

    def __getitem__(self, idx):
        score = self.scores.iloc[idx,:].to_numpy().astype('float32').reshape( (3,-1), order = 'A')
        gt = np.array([self.labels.iloc[idx]]).astype('int').reshape((-1,1))
        gt_one_hot = self.GTidx_to_rewards(gt).astype('int')

        # hypothesis of 
        # gt_hyp = np.array([self.idx['ground_truth'][idx]]).astype('int').reshape(-1,3)
        
        sample = {'scores': score, 'gt_labels': gt, 'gt_one_hot':gt_one_hot}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def GTidx_to_rewards(self, Y):
        
        newY = np.zeros((len(Y), self.Hall)).astype('float')
        for i in range(len(Y)):
            newY[i,Y[i]] = 1
        return newY

class ToTensor(object):
    """ Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        scores, gt_one_hot, gt_labels = sample['scores'], sample['gt_one_hot'], sample['gt_labels']

        return {'scores': torch.from_numpy(scores).type(torch.float32),
                'gt_one_hot':torch.from_numpy(gt_one_hot).type(torch.float32),
                'gt_labels': torch.from_numpy(gt_labels).type(torch.int32)}


def labels_to_hypothesis(labels, dataset):
    """
    Takes in labels and returns the hypothesis
    """
    # print('---labels:',labels.shape)
    labels = labels.reshape((len(labels),)) # ensure shape of labels is batchx1
    # print('---after:',labels.shape)
    Hall = dataset.Hall

    # get the order of hypotheses based on how they were stored
    index_df = dataset.idx.iloc[0][1:3*Hall+1].to_numpy()
    # print('\t index_df', len(index_df))

    # store all hypotheses in a numpy array
    hyp = np.zeros((len(labels),3), dtype = 'int')
    for i,l in enumerate(labels):
        # print('\t \t l:', l)
        hyp[i,:] = index_df[l*3:l*3+3]
    return hyp
    
def semER(hyp1, hyp2):
    """
    Calculates the Semantic Error Rate between two hypothesis.
    If the domains are not the same, returns dimensions of hypotheses
    Else: returns the Levenshtein dist.
    """

    domain_check = (hyp1[:,0] == hyp2[:,0]) # False if domain is not equal
    err = sum((1-domain_check)*len(hyp1[0])) # all dimensions need to be changed

    for i in range(1,len(hyp1[0])): # check for intent and entity
        err += sum((hyp1[:,i] != hyp2[:,i])*(domain_check)) # add error if domain is the same

    return err

def interER(hyp1, hyp2):
    """ Returns the interpretation error rate. 1 if the hypotheses are not the same"""

    return sum([0 if (hyp1[i]==hyp2[i]).all() else 1 for i in range(len(hyp1))])
