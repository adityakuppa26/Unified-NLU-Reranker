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

from utils.util import AnnotatedDataset, ToTensor, labels_to_hypothesis, semER, interER, random_split, load_config
from agents import LinUCB, LinearClassifier, DistributedLinearClassifier, nonLinearClassifier, ReinforceUnified, ReinforceDistributed
from utils.train_util import train_one_epoch, valid_one_epoch
# weights and biases
import wandb


######## Upload Raw Data #############

# run = wandb.init(project=PROJECT_NAME,
#                  entity = ENTITY_NAME, #entity is the team directory. Has to be made on wandb website first.
#                  job_type = 'upload_data', # every time we run wandb, we will define different job types so we can split our work
#                  config = {
#                       'dataset': 'dataset_2' #does not have to be the same name as data artifact
#                       }
#                  )
# # create a new artifact that has data-type "uncalibrated_data" and desired name
# # Every time we run this, if the same object already exists, it will just use that
# # again.
# my_data = wandb.Artifact("dataset_2", type = "uncalibrated_data") 
# my_data.add_file(SCORE_DATASET, name='scores')
# my_data.add_file(INDEX_DATSET, name='indices')
# run.log_artifact(my_data)
# run.finish()


# def train_one_epoch(agent, agent_type, dataloader,configs, optimizer=None, criterion=None, distributed = None, device = 'cpu'):
#     """
#     Trains one epoch. Currently can train agent types:
#       - linear classifier
#       - Linear UCB agent
#     """
#     running_loss = 0.0
    
#     # define hyper-params
#     # Hspace = configs.Hall if distributed == None else configs.Hspaces[distributed]
#     Hspace = configs.Hall
#     models_in_domain = configs.num_of_classifiers # 65 and 3

#     for batch_idx, sample_batched in enumerate(dataloader):
#         scores_batch = sample_batched['scores'].reshape(-1, Hspace*models_in_domain).to(device)
#         gt_one_hot = sample_batched['gt_one_hot'].reshape(-1,Hspace).to(device)
#         gt_labels = sample_batched['gt_labels'].to(device)
        
#         if agent_type == 'linear':
#             label = torch.tensor(gt_labels)
#             optimizer.zero_grad()
            
#             # forward + backward + optimize
#             outputs = agent(scores_batch)
#             loss = criterion(outputs, gt_one_hot)
#             loss.backward()
#             optimizer.step()
            
#             # update statistics
#             running_loss = loss.item()

#         # LinUCB
#         if agent_type == 'linUCB':
#             reward = agent.learn(scores_batch.reshape((-1,models_in_domain, Hspace)),
#                         gt_one_hot.reshape((-1, Hspace)))
            
#             # update statistics
#             running_loss += -reward # negative reward so that minimizing objective is equivalent

#     return running_loss

# def valid_one_epoch(agent, agent_type, dataloader, configs, distributed = None, dataset=None, device = 'cpu'):
#     """
#     Does inference on the model and returns the performance measures.
#     Make sure model has a .predict() method that returns a tensor of predicted hypothesis.
#     Measures returned so far:
#       - Semantic Error Rate
#       - Interpretation Error Rate
#     """
#     with torch.no_grad():
#         sem_error = 0; inter_error = 0; ctr = 0

#         # define hyper-params
#         # Hspace = configs.Hall if distributed == None else configs.Hspaces[distributed]
#         Hspace = configs.Hall
#         models_in_domain = configs.num_of_classifiers # 65 and 3

#         for _, sample_batched in enumerate(dataloader):

#             # forward + predicted hypothesis
#             scores_batch = sample_batched['scores'].reshape(-1, Hspace*models_in_domain).to(device)
#             outputs = agent.predict(scores_batch).detach().cpu().numpy()
#             pred_hyp = labels_to_hypothesis(outputs, dataset)

#             # ground truth hypotheses
#             gt_labels = sample_batched['gt_labels']
#             gt_hyp = labels_to_hypothesis(gt_labels, dataset)

#             # performance measures
#             sem_error += semER(pred_hyp, gt_hyp)
#             inter_error += interER(pred_hyp, gt_hyp)
#             # ctr += configs.batch_size
#             ctr += gt_hyp.shape[0]
#         # print('Validation epoch over. Counter:', ctr)
#         return (sem_error/ctr, inter_error/ctr)

######## Loading the annotated dataset ########

# dataset = AnnotatedDataset(score_csv = SCORE_DATASET,
#                                  idx_csv = INDEX_DATSET,
#                                  transform = ToTensor())

# login into wandb
wandb.login(key='063ba9d8afafed34c54d2a3b805fdc272e20ff7e')

# get config from YAML
CONFIG_PATH = "configs/"
config = load_config("wandb_reinforce_dist_large.yaml", CONFIG_PATH)

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# create train and validation data loaders
dataset = AnnotatedDataset(score_csv = config["SCORE_DATASET"],
                                idx_csv = config["INDEX_DATSET"],
                                transform = ToTensor(),
                                limit = config["subset_size"])

train_data, valid_data = random_split(dataset, config["parameters"]["train_valid"]["value"])
print('Configs:', config["parameters"]["train_valid"])

def main_hyp(train_data = train_data, valid_data = valid_data):
    """ Main Training Loop """
    # combined_config = config_meta_default | config_agent_default[agent_type]
    
    # sweep
    run = wandb.init(job_type = "sweep")

    config = wandb.config
    agent_type = config["agent_type"]
    framework = config["framework"]

    # create train and validation data loaders
    train_loader = DataLoader(train_data, batch_size = config["batch_size"],
                            shuffle = False, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size = config["batch_size"],
                            shuffle = False, num_workers=0)
    
    #instantiate models
    if agent_type == 'linear':
        if framework == 'dist':
            agent = DistributedLinearClassifier(Hspaces = config["Hspaces"],
                                            num_of_classifiers = config["num_of_classifiers"])
        elif framework == 'unif':
            agent = LinearClassifier(input_dim=config["Hall"] * config["num_of_classifiers"],
                                 num_of_classes = config["Hall"])
        else:
            agent = nonLinearClassifier(input_dim=config["Hall"] * config["num_of_classifiers"],
                                hidden_dim = config["hidden_dim"],
                                 num_of_classes = config["Hall"])
            
        optimizer = optim.SGD(agent.parameters(), lr = config["lr"], momentum = config["momentum"])
        criterion = nn.CrossEntropyLoss()
        
        # since these are nn.Module, we send them to GPU
        agent.to(device)


    if agent_type == 'linUCB':
        agent = LinUCB(alpha=config["lr"], k=config["Hall"], d=config["num_of_classifiers"], device = device)
        optimizer = None; criterion = None

    if agent_type == "reinforce":
            if framework =='unif':
                # Reinforce Unified
                agent = ReinforceUnified(config["epsilon"])
                optimizer = optim.SGD(agent.policy.parameters(), 
                                    lr=config["lr"],
                                    momentum = config["momentum"])
            elif framework == 'dist':
                agent = ReinforceDistributed(epsilon = config["epsilon"],
                                            Hspaces = config["Hspaces"],
                                            num_of_classifiers = config["num_of_classifiers"])
                optimizer = optim.SGD(agent.parameters(), 
                                    lr=config["lr"],
                                    momentum = config["momentum"])
            criterion = None

    # if torch.cuda.device_count() > 1:
    #     print("We will use", torch.cuda.device_count(), "GPUs!")
    #     agent = nn.DataParallel(agent)
    #     agent = agent.module
    
    # name_details = ''
    # for i in config_agent_default[agent_type]:
    #     name_details += i+':'+str(config_agent_default[agent_type][i])

    for epoch in range(config["epochs"]):
        # get matrics after each epoch of training and validating results.
        train_loss = train_one_epoch(agent = agent,
                                     agent_type = agent_type,
                                     dataloader = train_loader,
                                     config = config,
                                     optimizer = optimizer,
                                     criterion = criterion)
        (valid_sem_err, valid_inter_err) = valid_one_epoch(agent = agent,
                                              agent_type = agent_type,
                                              dataloader = valid_loader,
                                              config = config,
                                              dataset = dataset)
        # log the metrics
        run.log({
            'epoch':epoch,
            'train_loss':train_loss,
            'valid_sem_err':valid_sem_err,
            'valid_inter_err':valid_inter_err
        })

def main( train_data, test_data, final_config, runs = 5):
    """ Main Training Loop """
    
    seeds = final_config["seeds"]
    project_name = final_config["PROJECT_NAME"]
    entity = final_config["ENTITY"]
    for r in range(runs):
        np.random.seed(seeds[r])
        torch.manual_seed(seeds[r])

        # initialize a run
        run = wandb.init(job_type = "final_runs",
                         project = project_name,
                         entity = entity,
                         config = final_config)
        config = wandb.config
        print('Loop config:', config)
        # create train and validation data loaders
        train_loader = DataLoader(train_data, batch_size = config["batch_size"],
                                shuffle = True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size = config["batch_size"],
                                shuffle = True, num_workers=0)

        agent_type = config["agent_type"]
        framework = config["framework"]
    
        #instantiate models
        if agent_type == 'linear':
            if framework == 'dist':
                agent = DistributedLinearClassifier(Hspaces = config["Hspaces"],
                                                num_of_classifiers = config["num_of_classifiers"])
            elif framework == 'unif':
                agent = LinearClassifier(input_dim=config["Hall"] * config["num_of_classifiers"],
                                    num_of_classes = config["Hall"])
            else:
                agent = nonLinearClassifier(input_dim=config["Hall"] * config["num_of_classifiers"],
                                    hidden_dim = config["hidden_dim"],
                                    num_of_classes = config["Hall"])
                
            optimizer = optim.SGD(agent.parameters(), lr = config["lr"], momentum = config["momentum"])
            criterion = nn.CrossEntropyLoss()
            
            # since these are nn.Module, we send them to GPU
            agent.to(device)

        if agent_type == 'linUCB':
            agent = LinUCB(alpha=config["lr"], k=config["Hall"], d=config["num_of_classifiers"], device = device)
            optimizer = None; criterion = None

        if agent_type == "reinforce":
            if framework =='unif':
                # Reinforce Unified
                agent = ReinforceUnified(config["epsilon"])
                optimizer = optim.SGD(agent.policy.parameters(), 
                                    lr=config["lr"],
                                    momentum = config["momentum"])
            elif framework == 'dist':
                agent = ReinforceDistributed(epsilon = config["epsilon"],
                                            Hspaces = config["Hspaces"],
                                            num_of_classifiers = config["num_of_classifiers"])
                optimizer = optim.SGD(agent.parameters(), 
                                    lr=config["lr"],
                                    momentum = config["momentum"])
            criterion = None

        for epoch in range(config["epochs"]):
            # get matrics after each epoch of training and validating results.
            train_loss = train_one_epoch(agent = agent,
                                        agent_type = agent_type,
                                        dataloader = train_loader,
                                        config = config,
                                        optimizer = optimizer,
                                        criterion = criterion)
            (valid_sem_err, valid_inter_err) = valid_one_epoch(agent = agent,
                                                agent_type = agent_type,
                                                dataloader = test_loader,
                                                config = config,
                                                dataset = dataset)
            # log the metrics
            run.log({
                'epoch':epoch,
                'train_loss':train_loss,
                'valid_sem_err':valid_sem_err,
                'valid_inter_err':valid_inter_err
            })
        run.finish()

if __name__=="__main__":
    for i in range(len(dataset)):
        sample = dataset[i]
        if i <5:
            print('i:',i,'scores:', sample['scores'].dtype, '\n gt:', sample['gt_labels'].shape)
    existing_sweep = ''
    # sweep_id = wandb.sweep(sweep = config) if existing_sweep == '' else existing_sweep

    # run the sweep
    # wandb.agent(sweep_id, function = main_hyp)



    # get config from YAML
    CONFIG_PATH = "configs/"
    final_config = load_config("final_reinforce_dist.yaml", CONFIG_PATH)
    dataset = AnnotatedDataset(score_csv = final_config["SCORE_DATASET"],
                                idx_csv = final_config["INDEX_DATSET"],
                                transform = ToTensor(),
                                limit = -final_config["subset_size"])

    train_data, test_data = random_split(dataset, final_config["train_test"])
    # final_config = [""]
    main(train_data=train_data, test_data=test_data, final_config = final_config)

# linear_sweep_config = {
#     'project' : PROJECT_NAME,
#     'entity' : ENTITY_NAME,
#     'method': 'grid',
#     'name': 'full_dist_linear',
#     'metric': {'goal':'minimize', 'name':'valid_sem_err'},
#     'parameters':{
#         "Hall": {'value': dataset.Hall},  "num_of_classifiers": {'value':3},
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
# wandb_configs = {
#     "Hall":sum(Hspaces), "num_of_classifiers":3,
#     "Hspaces":Hspaces,
#     "d":len(I), "I":I, "E":E,
#     "epochs":100,
#     "train_valid_test":[0.65, 0.15, 0.2],
#     "verbose_freq":3,
#     "momentum":0.6,
#     "hidden_dim":200,
#     "batch_size":200,
#     "lr":1e-4
# }
# existing_sweep = ''
# sweep_id = wandb.sweep(sweep = linear_sweep_config) if existing_sweep == '' else existing_sweep
# print('sweep id:', sweep_id, type(sweep_id))

# # run the sweep
# wandb.agent(sweep_id, function = main)

# wandb.login(key='063ba9d8afafed34c54d2a3b805fdc272e20ff7e')
# PROJECT_NAME = "final_hyper_search"
# ENTITY_NAME = "amazon_alexa"
# SCORE_DATASET = 'datasets/unit_test_uncal_score.csv'
# INDEX_DATSET = 'datasets/unit_test_hyp_idx.csv'

# W&B Sweep
# I = [5, 7, 3]; E = [10, 5, 15]
# I = [2,3]; E = [3,2]
# Hspaces = [0]+[I[i]*E[i] for i in range(len(I))]

# linear_sweep_config = {
#     'project' : PROJECT_NAME,
#     'entity' : ENTITY_NAME,
#     'method': 'grid',
#     'name': 'full_dist_linear',
#     'metric': {'goal':'minimize', 'name':'valid_sem_err'},
#     'parameters':{
#         "Hall": {'value': dataset.Hall},  "num_of_classifiers": {'value':3},
#         "Hspaces": {'value': Hspaces},
#         "d": {'value':2}, "I":{'value':[2,3]}, "E":{'value':[3,2]},
#         "train_valid_test":{'value':[0.65, 0.15, 0.2]},
#         "epochs": {'value':1000},
#         "verbose_freq": {'value':3},
#         "momentum": {'value': 0.7},
#         "hidden_dim": {'value': 200},
#         # "batch_size": {'value': 250},
#         # "lr": {'value': 1e-3}

#         "batch_size": {'values': [120, 200]},
#         "lr": {'values':[1e-2,1e-3, 1e-4]},
#      }
# }
