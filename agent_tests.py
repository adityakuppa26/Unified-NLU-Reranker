import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

from utils.util import AnnotatedDataset, ToTensor, labels_to_hypothesis, semER, interER, random_split
from agents import LinUCB, LinearClassifier, DistributedLinearClassifier, nonLinearClassifier
from train import train_one_epoch, valid_one_epoch

# weights and biases
import wandb
wandb.login(key='063ba9d8afafed34c54d2a3b805fdc272e20ff7e')
PROJECT_NAME = "LinUCB_tests"
ENTITY_NAME = "amazon_alexa"
SCORE_DATASET = 'datasets/data_2_uncal_score.csv'
INDEX_DATSET = 'datasets/data_2_hyp_idx.csv'


def overfit_test(agent_type='linUCB', framework='dist', num_of_datapoints=5, configs=None):
    """ Training Loop for overfitting over a few datapoints. """

    run = wandb.init(
        project = PROJECT_NAME,
        job_type = "overfit_test",
        config = configs,
        mode = "online"
    )
    cfg = wandb.config

    dataset = AnnotatedDataset(score_csv = SCORE_DATASET,
                                idx_csv = INDEX_DATSET,
                                transform = ToTensor(),
                                limit = num_of_datapoints)
    
    train_loader = DataLoader(dataset, batch_size = num_of_datapoints,
                              shuffle = True)
    
    #instantiate model
    if agent_type == 'linear':
        if framework == 'dist':
            agent = DistributedLinearClassifier(Hspaces = cfg.Hspaces,
                                            num_of_classifiers = cfg.num_of_classifiers)
        elif framework == 'unif':
            agent = LinearClassifier(input_dim=cfg.Hall * cfg.num_of_classifiers,
                                 num_of_classes = cfg.Hall)
        else:
            agent = nonLinearClassifier(input_dim=cfg.Hall * cfg.num_of_classifiers,
                                hidden_dim = cfg.hidden_dim,
                                 num_of_classes = cfg.Hall)
    
        optimizer = optim.SGD(agent.parameters(), lr = cfg.lr, momentum = cfg.momentum)
        criterion = nn.CrossEntropyLoss()
        
        # since these are nn.Module, we send them to GPU
        agent.to(device)

    if agent_type == 'linUCB':
        agent = LinUCB(alpha=cfg.lr, k=cfg.Hall, d=cfg.num_of_classifiers, device = device)
        optimizer = None; criterion = None

    for epoch in range(cfg.epochs):
        # get matrics after each epoch of training and validating results.
        train_loss = train_one_epoch(agent = agent,
                                     agent_type = agent_type,
                                     dataloader = train_loader,
                                     configs = cfg,
                                     optimizer = optimizer,
                                     criterion = criterion,
                                     device = device)
        
        (valid_sem_err, valid_inter_err) = valid_one_epoch(agent = agent,
                                              agent_type = agent_type,
                                              dataloader = train_loader,
                                              configs = cfg,
                                              dataset = dataset,
                                              device = device)
        
        # log the metrics
        run.log({
            'epoch':epoch,
            'train_loss':train_loss,
            'valid_sem_err':valid_sem_err,
            'valid_inter_err':valid_inter_err
        })

I = [5, 7, 3]; E = [10, 5, 15]
Hspaces = [0]+[I[i]*E[i] for i in range(len(I))]

wandb_configs = {
    "Hall":sum(Hspaces), "num_of_classifiers":3,
    "Hspaces":Hspaces,
    "d":len(I), "I":I, "E":E,
    "epochs":100,
    "train_valid_test":[0.65, 0.15, 0.2],
    "verbose_freq":3,
    "momentum":0.6,
    "hidden_dim":200,
    "batch_size":250,
    "lr":2e-4
}

if __name__ == "__main__":
    agent_type = 'linUCB'
    framework = 'dist'
    num_of_datapoints = 5
    print('Running overfit test on ', framework, 're-ranker of type:', agent_type, 'with', num_of_datapoints, 'points.')
    overfit_test(agent_type=agent_type,
                 framework=framework,
                 num_of_datapoints=num_of_datapoints,
                 configs=wandb_configs)
    