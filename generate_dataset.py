import pandas as pd
import numpy as np
import itertools as it

from utils.gen_dataset import load_config, domain_scorer, intent_entity_scorer, test_dataset, softmax, ECE_to_T

def gen_ground_truths(config):
    D = config["D"]
    I = config["I"]
    E = config["E"]

    dataset_size = config["dataset_size"]

    # generating labels for domain (how many would be of domain 1 and how many of domain 2)
    d_label = list(np.random.randint(1, D+1, dataset_size))

    domains = [[i] for i in range(1,D+1)]
    hypo_labels = pd.DataFrame(columns = ['DL', 'IL', 'EL'])
    d_s = []
    i_s = []
    e_s = []

    for i in range(D): # i = 0, 1
        num_examples = d_label.count(i+1) # gives how many random examples we want to generate based on how many examples of a domain are present in d_labels
        i_s.extend(list(np.random.randint(1, I[i]+1, num_examples))) #generating intent labels
        e_s.extend(list(np.random.randint(1, E[i]+1, num_examples))) #generating entity labels
        d_s.extend([i+1] * num_examples)

    hypo_labels['DL'] = d_s
    hypo_labels['IL'] = i_s
    hypo_labels['EL'] = e_s

    return hypo_labels

def gen_data_with_acc(hypo_labels, config):
    D = config["D"]
    I = config["I"]
    E = config["E"]
    d_acc = config["d_acc"]
    i_acc = config["i_acc"]
    e_acc = config["e_acc"]
    ground_truths = hypo_labels[['DL', 'IL', 'EL']].reset_index(drop=True)


    # Create the Dataframe
    u_dataset = pd.DataFrame(columns=['ground_truth'])
    multipliers = D
    for i in range(1, multipliers+1):
        u_dataset[f'd{i}'] = u_dataset['ground_truth'] * i

    for i in range(1, multipliers+1):
        u_dataset[f'i{i}'] = u_dataset['ground_truth'] * i

    for i in range(1, multipliers+1):
        u_dataset[f'e{i}'] = u_dataset['ground_truth'] * i

    # Generate the dataset
    for i in range(len(ground_truths)):
        l1 = []

        dl = ground_truths['DL'][i]
        il = ground_truths['IL'][i]
        el = ground_truths['EL'][i]

        dom_scores = domain_scorer(dl, d_acc)
        int_scores = intent_entity_scorer(dl, il, i_acc, I, 'random')
        ent_scores = intent_entity_scorer(dl, el, e_acc, E, 'random')

        l2 = []
        l2.append(dl)
        l2.append(il)
        l2.append(el)
        l1.append(l2)

        for j in range(D):
            l1.append(dom_scores[j])

        for j in range(D):
            l1.append(int_scores[j])

        for j in range(D):
            l1.append(ent_scores[j])
        if (i+1)%1000 == 0:
            print('generated data point:', i)

        u_dataset.loc[len(u_dataset)] = l1
    
    return u_dataset

def add_ECE_errors(calibrated_dataset,config):

    # ECE values
    d_ece = config["d_ece"]
    i_ece = config["i_ece"]
    e_ece = config["e_ece"]
    T = {}

    uncal_dataset = calibrated_dataset.copy()
    # loop over each model (d1, d2, ... etc) in the dataset
    for col in uncal_dataset:
        if col != 'ground_truth':
            if col[0] == 'd':
                T[col] = d_ece[int(col[-1])-1]
            elif col[0] == 'i':
                T[col] = i_ece[int(col[-1])-1]
            else:
                T[col] = e_ece[int(col[-1])-1]

            T[col] = ECE_to_T(T[col])
            
            # scale the scores using temperature scaling for each data point
            for i in range(0, len(uncal_dataset)):
                uncal_dataset[col][i] = softmax(np.log(np.array(calibrated_dataset[col][i])), T[col])
    print('Temperature scales:', T)
    return uncal_dataset

def cartesian_product(uncal_dataset,configs):
    D = configs["D"]; I = configs["I"]; E = configs["E"]
    Hall = sum([I[i]*E[i] for i in range(D)])
    numpy_dataset = np.zeros(shape = (len(uncal_dataset),3*Hall)) #stores the scores of hypotheses
    hyp_dataset = np.zeros(shape=(len(uncal_dataset), 3*Hall)) #stores the indices representing of hypotheses

    # gt_idx the index of the correct hypothesis. This helps create one-hot vectors for rewards.
    gt_idx = np.zeros(shape=(len(uncal_dataset),)) 

    for idx in range(len(uncal_dataset)):
        data_point = []; hyp_point = [] # store the scores and hypothesis idx

        for d in range(D): # since it is a restricted cartesian product, each is created separately
            d_j = 'd'+str(d+1); i_j = 'i'+str(d+1); e_j = 'e'+str(d+1)

            # get all scores for the data point (domain: scalar, intent and entity: arrays)
            scores = [uncal_dataset[d_j][idx], uncal_dataset[i_j][idx], uncal_dataset[e_j][idx]]
            hypothesis = [[d+1], list(range(1,I[d]+1)),  list(range(1,E[d]+1))] 
            p = it.product(*scores); hyp_p = it.product(*hypothesis) # cartesian product of scores and hypothesis index
            
            for i, (v, hyp_v) in enumerate(zip(p, hyp_p)):
                data_point += v # stacking each score (of type [d, i, e])
                if list(hyp_v) == uncal_dataset['ground_truth'][idx]:
                    # print('index of hypothesis:',hyp_v, i,[intent_dims[j]*entity_dims[j] for j in range(d)])
                    gt_idx[idx] = i + sum([I[j]*E[j] for j in range(d)])
                hyp_point += hyp_v
        
        numpy_dataset[idx] = data_point
        hyp_dataset[idx] = hyp_point

    newDF = pd.DataFrame(numpy_dataset)

    idxDF = pd.DataFrame(hyp_dataset)
    # idxDF = {}
    idxDF['ground_truth'] = uncal_dataset['ground_truth']
    idxDF['gt_idx'] = gt_idx
    # idxDF = pd.DataFrame(data = idxDF)
    
    return newDF, idxDF

if __name__=="__main__":
    # folder to load config file
    CONFIG_PATH = "configs/"

    # get config from YAML
    config = load_config("dataset_large.yaml", CONFIG_PATH)

    # generate ground truths
    hypothesis_labels = gen_ground_truths(config)

    # generate calibrated dataset
    calibrated_dataset = gen_data_with_acc(hypothesis_labels, config)
    print(calibrated_dataset.head(10))

    #test dataset generated
    test_dataset(calibrated_dataset, config, classifier_type='domain')
    test_dataset(calibrated_dataset, config, classifier_type='intent')
    test_dataset(calibrated_dataset, config, classifier_type='entity')

    uncalibrated_dataset = add_ECE_errors(calibrated_dataset, config)
    #test dataset generated
    test_dataset(uncalibrated_dataset, config, classifier_type='domain')
    test_dataset(uncalibrated_dataset, config, classifier_type='intent')
    test_dataset(uncalibrated_dataset, config, classifier_type='entity')
    # print(uncalibrated_dataset,'\n ------^^ Uncalibrated dataset ------')

    scores_dataset, idx_dataset = cartesian_product(uncalibrated_dataset, config)
    # print('Scores dataset:', scores_dataset, '\n index datasets:', idx_dataset)
    scores_dataset.to_csv(config["save-scores"], index=False)
    idx_dataset.to_csv(config["save-idx"])

