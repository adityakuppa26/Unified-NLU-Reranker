import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)

from util import AnnotatedDataset, ToTensor, labels_to_hypothesis, semER, interER, random_split
from agents import LinUCB, LinearClassifier, DistributedLinearClassifier, nonLinearClassifier

# weights and biases
import wandb

wandb.login(key='0ee6ef2a7a617bf5fd8305e7afd00eabb77cd42f')
PROJECT_NAME = "linear_unified"
ENTITY_NAME = "amazon_alexa"
SCORE_DATASET = '/Users/shatabdibhise/Documents/Semester-IV/696DS/Datasets/test_uncal_score_100k.csv'
INDEX_DATSET = '/Users/shatabdibhise/Documents/Semester-IV/696DS/Datasets/test_hyp_idx_100k.csv'


def train_one_epoch(agent, agent_type, dataloader, configs, optimizer=None, criterion=None, distributed=None):
    """
    Trains one epoch. Currently can train agent types:
      - linear classifier
      - Linear UCB agent
    """
    running_loss = 0.0

    # define hyper-params
    # Hspace = configs.Hall if distributed == None else configs.Hspaces[distributed]
    Hspace = configs.Hall
    models_in_domain = configs.num_of_classifiers  # 65 and 3

    sem_error = 0
    ctr = 0

    for batch_idx, sample_batched in enumerate(dataloader):
        scores_batch = sample_batched['scores'].reshape(-1, Hspace * models_in_domain).to(device)
        gt_one_hot = sample_batched['gt_one_hot'].reshape(-1, Hspace).to(device)
        gt_labels = sample_batched['gt_labels'].to(device)

        if agent_type == 'linear':
            label = torch.tensor(gt_labels)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = agent(scores_batch)
            loss = criterion(outputs, gt_one_hot)
            loss.backward()
            optimizer.step()

            # update statistics
            running_loss = loss.item()

            o = agent.predict(scores_batch).detach().cpu().numpy()
            pred_hyp = labels_to_hypothesis(o, trial_dataset)
            gt_hyp = labels_to_hypothesis(gt_labels, trial_dataset)
            sem_error += semER(pred_hyp, gt_hyp)
            #ctr += configs.batch_size
            ctr += gt_hyp.shape[0]



        # LinUCB
        if agent_type == 'linUCB':
            reward = agent.learn(scores_batch.reshape((-1, models_in_domain, Hspace)),
                                 gt_one_hot.reshape((-1, Hspace)))

            # update statistics
            running_loss += -reward  # negative reward so that minimizing objective is equivalent

    return (running_loss, sem_error/ctr)


def valid_one_epoch_loss(agent, agent_type, dataloader, configs, optimizer=None, criterion=None, distributed=None):

        """
        Does inference on the model and returns the performance measures.
        Make sure model has a .predict() method that returns a tensor of predicted hypothesis.
        Measures returned so far:
          - Semantic Error Rate
          - Interpretation Error Rate
        """
        sem_error = 0;
        inter_error = 0;
        ctr = 0
        runnin_loss = 0.0

        # define hyper-params
        # Hspace = configs.Hall if distributed == None else configs.Hspaces[distributed]
        Hspace = configs.Hall
        models_in_domain = configs.num_of_classifiers  # 65 and 3

        for _, sample_batched in enumerate(dataloader):
            # forward + predicted hypothesis
            scores_batch = sample_batched['scores'].reshape(-1, Hspace * models_in_domain).to(device)
            gt_one_hot = sample_batched['gt_one_hot'].reshape(-1, Hspace).to(device)
            outputs = agent.predict(scores_batch).detach().cpu().numpy()

            pred_hyp = labels_to_hypothesis(outputs, trial_dataset)

            optimizer.zero_grad()
            o = agent(scores_batch)
            loss = criterion(o, gt_one_hot)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()

            # ground truth hypotheses
            gt_labels = sample_batched['gt_labels']
            gt_hyp = labels_to_hypothesis(gt_labels, trial_dataset)

            # performance measures
            sem_error += semER(pred_hyp, gt_hyp)
            inter_error += interER(pred_hyp, gt_hyp)
            # ctr += configs.batch_size
            ctr += gt_hyp.shape[0]
        return (sem_error / ctr, inter_error / ctr, running_loss)


def valid_one_epoch(agent, agent_type, dataloader, configs, optimizer=None, criterion=None, distributed=None):
    """
    Does inference on the model and returns the performance measures.
    Make sure model has a .predict() method that returns a tensor of predicted hypothesis.
    Measures returned so far:
      - Semantic Error Rate
      - Interpretation Error Rate
    """
    with torch.no_grad():
        sem_error = 0;
        inter_error = 0;
        ctr = 0

        # define hyper-params
        # Hspace = configs.Hall if distributed == None else configs.Hspaces[distributed]
        Hspace = configs.Hall
        models_in_domain = configs.num_of_classifiers  # 65 and 3

        for _, sample_batched in enumerate(dataloader):
            # forward + predicted hypothesis
            scores_batch = sample_batched['scores'].reshape(-1, Hspace * models_in_domain).to(device)
            outputs = agent.predict(scores_batch).detach().cpu().numpy()

            pred_hyp = labels_to_hypothesis(outputs, trial_dataset)

            # ground truth hypotheses
            gt_labels = sample_batched['gt_labels']
            gt_hyp = labels_to_hypothesis(gt_labels, trial_dataset)

            # performance measures
            sem_error += semER(pred_hyp, gt_hyp)
            inter_error += interER(pred_hyp, gt_hyp)
            #ctr += configs.batch_size
            ctr += gt_hyp.shape[0]
        return (sem_error / ctr, inter_error / ctr)


######## Loading the annotated dataset ########

trial_dataset = AnnotatedDataset(score_csv=SCORE_DATASET,
                                 idx_csv=INDEX_DATSET,
                                 transform=ToTensor())


def main(agent_type='linear', framework='unif'):
    """ Main Training Loop """
    # combined_config = config_meta_default | config_agent_default[agent_type]
    run = wandb.init(job_type="sweep")
    cfg = wandb.config

    # create train and validation data loaders
    trial_dataset = AnnotatedDataset(score_csv=SCORE_DATASET,
                                     idx_csv=INDEX_DATSET,
                                     transform=ToTensor())
    train_data, valid_data, test_data = random_split(trial_dataset, cfg.train_valid_test)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0)

    # instantiate models
    if agent_type == 'linear':
        if framework == 'dist':
            agent = DistributedLinearClassifier(Hspaces=cfg.Hspaces,
                                                num_of_classifiers=cfg.num_of_classifiers)
        elif framework == 'unif':
            agent = LinearClassifier(input_dim=cfg.Hall * cfg.num_of_classifiers,
                                     num_of_classes=cfg.Hall)
        else:
            agent = nonLinearClassifier(input_dim=cfg.Hall * cfg.num_of_classifiers,
                                        hidden_dim=cfg.hidden_dim,
                                        num_of_classes=cfg.Hall)

        optimizer = optim.SGD(agent.parameters(), lr=cfg.lr, momentum=cfg.momentum)
        criterion = nn.CrossEntropyLoss()

        # since these are nn.Module, we send them to GPU
        agent.to(device)

    if agent_type == 'linUCB':
        agent = LinUCB(alpha=cfg.lr, k=cfg.Hall, d=cfg.num_of_classifiers)
        optimizer = None;
        criterion = None

    # if torch.cuda.device_count() > 1:
    #     print("We will use", torch.cuda.device_count(), "GPUs!")
    #     agent = nn.DataParallel(agent)
    #     agent = agent.module

    # name_details = ''
    # for i in config_agent_default[agent_type]:
    #     name_details += i+':'+str(config_agent_default[agent_type][i])

    print("Learning rate: ", cfg.lr)
    for epoch in range(cfg.epochs):
        # get matrics after each epoch of training and validating results.
        (train_loss, train_sem_err) = train_one_epoch(agent=agent,
                                     agent_type=agent_type,
                                     dataloader=train_loader,
                                     configs=cfg,
                                     optimizer=optimizer,
                                     criterion=criterion)

        (valid_sem_err, valid_inter_err, val_loss) = valid_one_epoch_loss(agent=agent,
                                                           agent_type=agent_type,
                                                           dataloader=valid_loader,
                                                           configs=cfg,
                                                           optimizer=optimizer,
                                                           criterion=criterion)
        # log the metrics
        run.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': val_loss,
            'valid_sem_err': valid_sem_err,
            'valid_inter_err': valid_inter_err,
            'train_sem_err': train_sem_err
        })


# W&B Sweep
#I = [5, 7, 3];
#E = [10, 5, 15]
I = [3, 5]
E = [10, 7]
Hspaces = [0] + [I[i] * E[i] for i in range(len(I))]
# linear_sweep_config = {
#     'project' : PROJECT_NAME,
#     'entity' : ENTITY_NAME,
#     'method': 'grid',
#     'name': 'full_dist_linear',
#     'metric': {'goal':'minimize', 'name':'valid_sem_err'},
#     'parameters':{
#         "Hall": {'value': trial_dataset.Hall},  "num_of_classifiers": {'value':3},
#         "Hspaces": {'value': Hspaces},
#         "d": {'value':3}, "I":{'value':[5, 7, 3]}, "E":{'value':[10, 5, 15]},
#         "train_valid_test":{'value':[0.65, 0.15, 0.2]},
#         "epochs": {'value':100},
#         "verbose_freq": {'value':3},
#         "momentum": {'value': 0.6},
#         "hidden_dim": {'value': 200},

#         "batch_size": {'values': [120, 200, 250]},
#         "lr": {'values':[1e-2, 1e-3, 1e-4, 1e-5]},
#      }
# }
linear_sweep_config = {
    'project': PROJECT_NAME,
    'entity': ENTITY_NAME,
    'method': 'grid',
    'name': 'full_dist_linear',
    'metric': {'goal': 'minimize', 'name': 'valid_sem_err'},
    'parameters': {
        "Hall": {'value': trial_dataset.Hall}, "num_of_classifiers": {'value': 3},
        "Hspaces": {'value': Hspaces},
        "d": {'value': 2}, "I": {'value': [3, 5]}, "E": {'value': [10, 7]},
        "train_valid_test": {'value': [0.65, 0.15, 0.2]},
        "epochs": {'value': 5000},
        "verbose_freq": {'value': 3},
        "momentum": {'value': 0.6},
        "hidden_dim": {'value': 200},
        "batch_size": {'value': 250},
        # "lr": {'value': 1e-3}

        # "batch_size": {'values': [120, 200, 250]},
        "lr": {'values': [0.01]},
    }
}
# existing_sweep = ''
# sweep_id = wandb.sweep(sweep = linear_sweep_config) if existing_sweep == '' else existing_sweep
# print('sweep id:', sweep_id, type(sweep_id))

# # run the sweep
# wandb.agent(sweep_id, function = main)

if __name__ == "__main__":
    for i in range(len(trial_dataset)):
        # print(i)
        sample = trial_dataset[i]
        if i < 5:
            print('i:', i, 'scores:', sample['scores'].dtype, '\n gt:', sample['gt_labels'].shape)
    existing_sweep = ''
    sweep_id = wandb.sweep(sweep=linear_sweep_config) if existing_sweep == '' else existing_sweep
    print('sweep id:', sweep_id, type(sweep_id))

    # run the sweep
    wandb.agent(sweep_id, function=main)
