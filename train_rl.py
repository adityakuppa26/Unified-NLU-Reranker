import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:',device)

import numpy as np
import pandas as pd
# import ast
# import matplotlib.pyplot as plt
# from IPython import display
# %matplotlib inline
np.set_printoptions(suppress=True)

from util import AnnotatedDataset, ToTensor, labels_to_hypothesis, semER, interER, random_split
from agents import LinUCB, LinearClassifier, DistributedLinearClassifier, nonLinearClassifier

# weights and biases
import wandb
wandb.login(key='063ba9d8afafed34c54d2a3b805fdc272e20ff7e')
PROJECT_NAME = "dataset_2_dist"
ENTITY_NAME = "amazon_alexa"
SCORE_DATASET = 'datasets/data_2_uncal_score.csv'
INDEX_DATSET = 'datasets/data_2_hyp_idx.csv'


def train_one_epoch_reinforce(agent,policy_type,dataloader, configs,optimizer, criterion):

    running_loss = 0.0

    # define hyper-parameters
    Hspace  = configs.Hall
    models_in_domain = configs.num_of_classifiers

    for _, sample_batched in enumerate(dataloader):
        scores_batch = sample_batched['scores'].reshape(-1, Hspace*models_in_domain).to(device)
        gt_one_hot = sample_batched['gt_one_hot'].reshape(-1,Hspace).to(device)
        gt_labels = sample_batched['gt_labels'].to(device)
        
        # if policy_type == 'softmax':
        #     for x,y in zip(scores_batch,)

######## Loading the annotated dataset ########

dataset = AnnotatedDataset(score_csv = SCORE_DATASET,
                                 idx_csv = INDEX_DATSET,
                                 transform = ToTensor())

def reinforce(policy_type='linear', framework='unif', configs=None):
    """ Main Training Loop """
    # combined_config = config_meta_default | config_agent_default[agent_type]
    
    # sweep
    # run = wandb.init(job_type = "sweep")

    # No sweep
    run = wandb.init(
        project = PROJECT_NAME,
        job_type = "overfit_test",
        config = configs
    )
    cfg = wandb.config

    # create train and validation data loaders
    dataset = AnnotatedDataset(score_csv = SCORE_DATASET,
                                 idx_csv = INDEX_DATSET,
                                 transform = ToTensor())
    train_data, valid_data, test_data = random_split(dataset, cfg.train_valid_test)
    train_loader = DataLoader(train_data, batch_size = cfg.batch_size,
                            shuffle = True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size = cfg.batch_size,
                            shuffle = True, num_workers=0)
    
    #instantiate models
    if policy_type == 'softmax':
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

    # if torch.cuda.device_count() > 1:
    #     print("We will use", torch.cuda.device_count(), "GPUs!")
    #     agent = nn.DataParallel(agent)
    #     agent = agent.module
    
    # name_details = ''
    # for i in config_agent_default[agent_type]:
    #     name_details += i+':'+str(config_agent_default[agent_type][i])

    for epoch in range(cfg.epochs):
        # get matrics after each epoch of training and validating results.
        train_loss = train_one_epoch_reinforce(agent = agent,
                                     policy_type = policy_type,
                                     dataloader = train_loader,
                                     configs = cfg,
                                     optimizer = optimizer,
                                     criterion = criterion)
        (valid_sem_err, valid_inter_err) = valid_one_epoch_reinforce(agent = agent,
                                              agent_type = agent_type,
                                              dataloader = valid_loader,
                                              configs = cfg,
                                              dataset = dataset)
        # log the metrics
        run.log({
            'epoch':epoch,
            'train_loss':train_loss,
            'valid_sem_err':valid_sem_err,
            'valid_inter_err':valid_inter_err
        })


# W&B Sweep
I = [5, 7, 3]; E = [10, 5, 15]
Hspaces = [0]+[I[i]*E[i] for i in range(len(I))]

linear_sweep_config = {
    'project' : PROJECT_NAME,
    'entity' : ENTITY_NAME,
    'method': 'grid',
    'name': 'full_dist_linear',
    'metric': {'goal':'minimize', 'name':'valid_sem_err'},
    'parameters':{
        "Hall": {'value': dataset.Hall},  "num_of_classifiers": {'value':3},
        "Hspaces": {'value': Hspaces},
        "d": {'value':3}, "I":{'value':[5, 7, 3]}, "E":{'value':[10, 5, 15]},
        "train_valid_test":{'value':[0.65, 0.15, 0.2]},
        "epochs": {'value':1000},
        "verbose_freq": {'value':3},
        "momentum": {'value': 0.6},
        "hidden_dim": {'value': 200},
        "batch_size": {'value': 250},
        # "lr": {'value': 1e-3}

        # "batch_size": {'values': [120, 200, 250]},
        "lr": {'values':[1e-3, 1e-4]},
     }
}
wandb_configs = {
    "Hall":sum(Hspaces), "num_of_classifiers":3,
    "Hspaces":Hspaces,
    "d":len(I), "I":I, "E":E,
    "epochs":100,
    "train_valid_test":[0.65, 0.15, 0.2],
    "verbose_freq":3,
    "momentum":0.6,
    "hidden_dim":200,
    "batch_size":200,
    "lr":1e-4
}
# existing_sweep = ''
# sweep_id = wandb.sweep(sweep = linear_sweep_config) if existing_sweep == '' else existing_sweep
# print('sweep id:', sweep_id, type(sweep_id))

# # run the sweep
# wandb.agent(sweep_id, function = main)

if __name__=="__main__":
    for i in range(len(dataset)):
        # print(i)
        sample = dataset[i]
        if i <5:
            print('i:',i,'scores:', sample['scores'].dtype, '\n gt:', sample['gt_labels'].shape)
    existing_sweep = ''
    # sweep_id = wandb.sweep(sweep = linear_sweep_config) if existing_sweep == '' else existing_sweep
    # print('sweep id:', sweep_id, type(sweep_id))

    # run the sweep
    # wandb.agent(sweep_id, function = main)

    main(agent_type='linear', framework='dist', configs=wandb_configs)