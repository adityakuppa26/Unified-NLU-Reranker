import numpy as np
import pandas as pd
# import ast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class LinUCB():
    def __init__(self, alpha, k, d=3, device='cpu'):
        self.alpha = alpha
        # num arms (hypothethes) for utterance
        self.k = k
        # d=3 (DC, IC, NER)
        self.d = d
        # A: will be covariance matrix??
        self.A = torch.eye(d)
        self.b = torch.zeros(d, 1)
        self.device = device
        self.A.to(self.device)
        self.b.to(self.device)

    def theta(self):
        theta = torch.matmul(torch.inverse(self.A), self.b) # 3x1
        return theta

    def get_ucbs(self, arms):
        """
            arms: batch x 3 x H_All
        """
        # print( self.theta().dtype,  arms.dtype)
        fin_scores = torch.matmul( arms, self.theta() ) # (100 x 6 x 3)*(3, 1) = #100 x 6 x 1
        ucbs = torch.matmul(arms,torch.inverse(self.A))
        ucbs = torch.sum(torch.mul(ucbs,arms), dim=-1, keepdim=True)
        ucbs = fin_scores + 1* torch.sqrt(ucbs)
        return ucbs

    def choose_arm(self, arms, ucbs):
        chosen_arm_idx = torch.argmax(ucbs, dim=1).squeeze() # 100x1
        # chosen_arm_idx = torch.tensor([7, 112, 118, 95, 79], dtype = torch.int)
        chosen_arm = torch.stack([arms[i,chosen_arm_idx[i],:] for i in range(len(chosen_arm_idx))]).reshape(-1, 3)
        return chosen_arm_idx, chosen_arm

    def get_reward(self, chosen_arm_idx, rewards):
        return torch.stack([rewards[i,chosen_arm_idx[i]] for i in range(len(chosen_arm_idx))])

    def update(self, chosen_arm, r):
        # print('\t Updating LinUCB. A was:', self.A, 'and b:', self.b)
        self.A += self.alpha*torch.matmul(chosen_arm, chosen_arm.T)
        self.b += self.alpha*chosen_arm * (r*2-1)
        # print('\t New A:', self.A, 'and b:', self.b)

    def learn(self, X, Y):
        avg_reward = 0
        X = X.to(self.device)
        Y = Y.to(self.device)
        X = X.reshape(-1, self.k, self.d); Y = Y.reshape(-1, self.k)
        ucbs = self.get_ucbs(arms= X) # 100x6x1
        chosen_arm_idx, chosen_arm = self.choose_arm(arms = X, ucbs=ucbs)
        r = self.get_reward(chosen_arm_idx, rewards=Y)
        for i in range(len(r)):
            self.update(chosen_arm[i,:].reshape(3,1)/len(r), r[i])
        return torch.mean(r).item()
    
    def predict(self, X):
        X = X.reshape(-1, self.k, self.d)
        ucbs = self.get_ucbs(arms= X)
        chosen_arm_idx, _ = self.choose_arm(arms = X, ucbs=ucbs) # 100x1 (or batch_sizex1)
        return chosen_arm_idx

class DistributedLinUCB():
    def __init__(self, alpha, Hspaces, d=3, device='cpu', domains=2):
        self.agents = []
        for i in range(domains):
            self.agents.append(LinUCB(alpha = alpha,
                                      k= Hspaces[i], d=d,
                                      device=device))
        self.score_splits = [sum(Hspaces[:i+1])*d for i in range(1,len(Hspaces)-1)]
        self.alpha = alpha
        # num arms (hypothethes) for utterance
        self.Hspaces = Hspaces
        # d=3 (DC, IC, NER)
        self.d = d
        self.device = device

    def choose_arm(self, arms, ucbs):
        chosen_arm_idx = torch.argmax(ucbs, dim=1).squeeze()
        chosen_arm = torch.stack([arms[i, chosen_arm_idx[i], :] for i in range(len(chosen_arm_idx))]).reshape(-1,3)

    def get_reward(self, chosen_arm_idx, rewards):
        return torch.stack([rewards[i,chosen_arm_idx[i]] for i in range(len(chosen_arm_idx))])

    def learn(self, X, Y):

        # split scores so that each domain agent only gets its own scores
        x_splits = torch.tensor_split(X, self.score_splits, dim=1)
        y_splits = torch.tensor_split(Y, self.score_splits, dim=1)
        
        r_total = 0; combined_ucbs = []
        for agent, x_i, y_i, hspace in zip(self.agents, x_splits, y_splits, self.Hspaces):
            x_i.reshape(-1, hspace, self.d); y_i = y_i.reshape(-1,hspace)

            ucbs = agent.get_ucbs(arms=x_i)
            combined_ucbs.append(ucbs)
            # chosen_arm_idx, chosen_arm = agent.choose_arm(arms = x_i, ucbs=ucbs)
            # r = agent.get_reward(chosen_arm_idx, rewards = y_i)
            # for i in range(len(r)):
            #     agent.update(chosen_arm[i,:].reshape(3,1)/len(r), r[i])
            # r_total += torch.mean(r).item()
        combined_ucbs = torch.cat(combined_ucbs, dim=1)
        chosen_arm_idx, chosen_arm = self.choose_arm(arms = X, ucbs = ucbs)
        r = self.get_reward(chosen_arm_idx, rewards = Y)
        
        return r_total

    def predict(self, X):
        X = X.reshape(-1, self.k, self.d)
        ucbs = self.get_ucbs(arms= X)
        chosen_arm_idx, _ = self.choose_arm(arms = X, ucbs=ucbs) # 100x1 (or batch_sizex1)
        return chosen_arm_idx

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_of_classes):
        super().__init__()
        self.Hall = num_of_classes
        self.linear = nn.Linear(input_dim, num_of_classes)

    def forward(self, x):
        x = self.linear(x)
        return x
    
    def predict(self,x):
        x = self.forward(x)
        return torch.argmax(x.reshape(-1,self.Hall), dim=1)
    
class nonLinearClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_of_classes):
        super().__init__()
        self.Hall = num_of_classes

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_of_classes)
        # self.mods = nn.ModuleList()
        # self.mods.append(self.relu)
        # self.mods.append(self.linear1)
        # self.mods.append(self.linear2)

    def forward(self, x):
        # print('x device:',x.get_device())
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def predict(self,x):
        x = self.forward(x)
        return torch.argmax(x.reshape(-1,self.Hall), dim=1)

class DistributedLinearClassifier(nn.Module):
    def __init__(self, Hspaces, num_of_classifiers):
        """
        Hspaces: array of shape D+1
                Hspaces[0] = 0 and for i>0 Hspaces[i] = number of hypotheses in a single domain
        num_of_classifers: for intent, entity and domain framework, this is equal to 3
        """
        super().__init__()
        self.Hall = sum(Hspaces)
        self.Hspaces = Hspaces
        self.score_splits = [sum(Hspaces[:i+1])*num_of_classifiers for i in range(1,len(Hspaces)-1)]
        self.num_of_classifiers = num_of_classifiers

        self.agents = nn.ModuleList()
        for d in range(1,len(Hspaces)):
            self.agents.append(nn.Linear(self.Hspaces[d]*self.num_of_classifiers, self.Hspaces[d]))
        # self.linear = nn.Linear(input_dim, num_of_classes)
    
    def forward(self, x):
        # split scores so that each domain agent only gets its own scores
        x_splits = torch.tensor_split(x, self.score_splits, dim=1)

        # forward pass for all agents
        y_i = []
        for agent, x_i in zip(self.agents, x_splits):
            y_i.append(agent(x_i))
        
        # concatenate distributed scores
        y = torch.cat(y_i, dim=1)
        return y
    
    def predict(self,x):
      	# make a forward pass and then returns the index of the highest rated score
        y = self.forward(x)
        return torch.argmax(y.reshape(-1,self.Hall), dim=1)


class ReinforceUnified(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        # self.policy = torch.nn.Sequential(
        #     torch.nn.Linear(3, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 1),
        #     torch.nn.Softmax(dim=0))

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(3, 1))
        
        self.softmax = torch.nn.Softmax(dim=1)

        self.batch_dist = None
        self.epsilon = epsilon
        
    # Input: batch x 3H
    # Returns: batch x 1 (idx of action taken)   
    def predict(self, X):
        # print('PREDICT')
        # reshape batch X (batch, 3H) to (batch, H, 3)
        obs = X.reshape((int(X.shape[0]), int(X.shape[1]/3), 3)).float()

        # print('Observations:')
        # print(obs[:4])

        scores = self.softmax(self.policy(obs)) # (batch, H, 1)

        # print('Softmax scores:')
        # print(scores[:4])

        dist = torch.distributions.Categorical(probs=scores.view(scores.size()[0], -1)) # dist.probs is (batch, H)

        self.batch_dist = dist

        # print('Categorical probs:')
        actions = dist.sample()
        best_actions = torch.argmax(scores, dim = 1) # the highest scored action

        # epsilon-exploration
        for i in range(scores.size()[0]):
            single_dist = torch.distributions.Categorical(probs = scores.view(scores.size()[0], -1)[i])
            for _ in range(self.epsilon):
                action = single_dist.sample() # (batch, 1), idx of action taken (chosen hypothethis)
                actions[i] = action
                if action == best_actions[i]:
                    break

        return actions 
    
    # Input: actions (batch x 1), Y as in LinUCB
    def get_rewards(self, actions, Y):
        ground_truth_idx = torch.argmax(Y, dim=1) # convert one-hot to idx (batch, 1)
        rewards = torch.where(actions == ground_truth_idx, torch.tensor(1), torch.tensor(-1)) # should be (batch, 1)
        
        return rewards
    
    def loss_batch(self, actions, rewards):
        log_prob = self.batch_dist.log_prob(actions) # (batch)
        loss = -log_prob * rewards # (batch)
        loss_batch = torch.mean(loss) # (1)

        return loss_batch


class ReinforceDistributed(nn.Module):
    def __init__(self, epsilon, Hspaces, num_of_classifiers):
        super().__init__()
        self.Hall = sum(Hspaces)
        self.Hspaces = Hspaces
        self.score_splits = [sum(Hspaces[:i+1])*num_of_classifiers for i in range(1,len(Hspaces)-1)]
        self.num_of_classifiers = num_of_classifiers

        # self.policy = torch.nn.Sequential(
        #     torch.nn.Linear(3, 1))
        
        # make one policy for each domain
        self.policies = nn.ModuleList()
        for d in range(1, len(Hspaces)):
            self.policies.append(nn.Linear(self.num_of_classifiers, 1))
        
        self.softmax = torch.nn.Softmax(dim=1)

        self.batch_dist = None
        self.epsilon = epsilon
        
    # Input: batch x 3H
    # Returns: batch x 1 (idx of action taken)   
    def predict(self, X):
        # split scores so that each domain agent only gets its own scores
        x_splits = torch.tensor_split(X, self.score_splits, dim=1)
        
        # forward pass for all agents
        scores_i = []
        for policy, X_i in zip(self.policies, x_splits):
            # reshape batch X (batch, 3H_i) to (batch, H_i, 3)
            obs = X_i.reshape((int(X_i.shape[0]), int(X_i.shape[1]/3), 3)).float()
            scores_i.append(self.softmax(policy(obs))) # (batch, H_i, 1)
            # print('-------X: {}, Obs: {}'.format(X_i.shape, obs.shape))
            throw = X_i.reshape((int(X_i.shape[0]), int(X_i.shape[1]/3), 3))
            # print('Reshaped:', throw[0,:,:])
            # print('OG:', X_i[0])
        # concatenate distributed scores
        scores = torch.cat(scores_i, dim=1)

        # create distributions using scores
        dist = torch.distributions.Categorical(probs=scores.view(scores.size()[0], -1)) # dist.probs is (batch, H)
        self.batch_dist = dist

        # print('Categorical probs:')
        actions = dist.sample()
        best_actions = torch.argmax(scores, dim = 1) # the highest scored action

        # # epsilon-exploration
        for i in range(scores.size()[0]):
            single_dist = torch.distributions.Categorical(probs = scores.view(scores.size()[0], -1)[i])
            for _ in range(self.epsilon):
                action = single_dist.sample() # (batch, 1), idx of action taken (chosen hypothethis)
                actions[i] = action
                if action == best_actions[i]:
                    break

        return actions 
    
    # Input: actions (batch x 1), Y as in LinUCB
    def get_rewards(self, actions, Y):
        ground_truth_idx = torch.argmax(Y, dim=1) # convert one-hot to idx (batch, 1)
        rewards = torch.where(actions == ground_truth_idx, torch.tensor(1), torch.tensor(-1)) # should be (batch, 1)
        
        return rewards
    
    def loss_batch(self, actions, rewards):
        log_prob = self.batch_dist.log_prob(actions) # (batch)
        loss = -log_prob * rewards # (batch)
        loss_batch = torch.mean(loss) # (1)

        return loss_batch