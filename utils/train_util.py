import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

from utils.util import labels_to_hypothesis, semER, interER

def train_one_epoch(agent, agent_type, dataloader,config, optimizer=None, criterion=None, distributed = None, device = 'cpu'):
    """
    Trains one epoch. Currently can train agent types:
      - linear classifier
      - Linear UCB agent
      - All other models: make it that the agent can output batch x 1 vector using a "predict"
                          function that takes batch x (Hall*3) input vector
    """
    running_loss = 0.0
    Hspace = config["Hall"]
    models_in_domain = config["num_of_classifiers"]

    for batch_idx, sample_batched in enumerate(dataloader):
        scores_batch = sample_batched['scores'].reshape(-1, Hspace*models_in_domain).to(device)
        gt_one_hot = sample_batched['gt_one_hot'].reshape(-1,Hspace).to(device)
        gt_labels = sample_batched['gt_labels'].to(device)
        
        if agent_type == 'linear':
            # label = torch.tensor(gt_labels)
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = agent(scores_batch)
            loss = criterion(outputs, gt_one_hot)
            loss.backward()
            optimizer.step()
            
            # update statistics
            running_loss = loss.item()

        # LinUCB
        elif agent_type == 'linUCB':
            reward = agent.learn(scores_batch.reshape((-1,models_in_domain, Hspace)),
                        gt_one_hot.reshape((-1, Hspace)))
            
            # update statistics
            running_loss += -reward # negative reward so that minimizing objective is equivalent

        # Reinforce
        elif agent_type == 'reinforce':
            # print("I'm here and I'm ReinforceUnified")
            optimizer.zero_grad()

            # # print poilcy weights
            # print('POLICY WEIGHTS')
            # for name, param in agent.policy.named_parameters():
            #      if param.requires_grad:
            #          print(name, param.data)
            
            outputs = agent.predict(scores_batch)
            rewards = agent.get_rewards(outputs, gt_one_hot.reshape((-1, Hspace)))
            loss = agent.loss_batch(actions = outputs, rewards = rewards)
            loss.backward()
            optimizer.step()

            # update statistics
            running_loss = loss.item()

        else:
            label = torch.tensor(gt_labels)
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = agent.predict(scores_batch)
            loss = criterion(outputs, gt_one_hot)
            loss.backward()
            optimizer.step()
            
            # update statistics
            running_loss = loss.item()

    return running_loss

def valid_one_epoch(agent, agent_type, dataloader, config, distributed = None, dataset=None, device = 'cpu'):
    """
    Does inference on the model and returns the performance measures.
    Make sure model has a .predict() method that returns a tensor of predicted hypothesis.
    Measures returned so far:
      - Semantic Error Rate
      - Interpretation Error Rate
    """
    sem_error = 0; inter_error = 0; ctr = 0
    Hspace = config["Hall"]
    models_in_domain = config["num_of_classifiers"]
    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):

            # forward + predicted hypothesis
            scores_batch = sample_batched['scores'].reshape(-1, Hspace*models_in_domain).to(device)
            outputs = agent.predict(scores_batch).detach().cpu().numpy()
            pred_hyp = labels_to_hypothesis(outputs, dataset)

            # ground truth hypotheses
            gt_labels = sample_batched['gt_labels']
            gt_hyp = labels_to_hypothesis(gt_labels, dataset)

            # performance measures
            sem_error += semER(pred_hyp, gt_hyp)
            inter_error += interER(pred_hyp, gt_hyp)
            # ctr += configs.batch_size
            ctr += gt_hyp.shape[0]
        # print('Validation epoch over. Counter:', ctr)
        return (sem_error/ctr, inter_error/ctr)