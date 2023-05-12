import pandas as pd
import numpy as np
import itertools as it
import yaml
import os

def load_config(config_name, CONFIG_PATH):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

# config = load_config("dataset.yaml")

def domain_scorer(dlabel, d_acc):
    """
    Generates domain scores to mimic their respective accuracies.
    Parameters:
        dlabel:     (int) ground truth domain label
        d_acc:      (array) containing accuracies of all classifiers of type intent/entity
    """
    # D = config["D"]
    D = len(d_acc)
    d_scores = [[]] * D
    for i in range(D):
    
        if i == dlabel-1:  # correct domain
            threshold = np.random.uniform(0, 1, 1)
            if threshold < d_acc[i]:    #TP
                score = d_acc[i]
            else:                   # FN
                score = np.around(1 - d_acc[i], 3)
        else:
            threshold = np.random.uniform(0, 1, 1)
            if threshold < d_acc[i]:    #TN
                score = np.around(1 - d_acc[i], 3)
            else:                   # FP
                score = d_acc[i]

        d_scores[i] = [score]

    return d_scores

def intent_entity_scorer(dlabel, clabel, acc, classifier, FN_option):
    """
    Generates intent/entity scores to mimic their respective accuracies.
    Parameters:
        dlabel:     (int) ground truth domain label
        clabel:     (int) ground truth class for intent/entity classifier of 'dlabel' domain
        acc:        (array) containing accuracies of all classifiers of type intent/entity
        classifier: (array) containing dimensions of all classifiers of type intent/entity
        FN_option:  (str) if random, then the accuracies are mimiced using a uniform distribution.
    """
    # D = config["D"]
    D = len(acc)
    scores = [[]] * D
    C = classifier

    for i in range(D):

        scores[i] = [0] * C[i]
        if i == dlabel - 1:     # correct domain
            threshold = np.random.uniform(0, 1, 1)
            if threshold < acc[i]:    #TP
                scores[i][clabel-1] = acc[i]
                scores[i][:clabel-1] = [np.around((1 - acc[i])/(C[i]-1), 3)] * len(scores[i][:clabel-1])
                scores[i][clabel:] = [np.around((1 - acc[i])/(C[i]-1), 3)] * len(scores[i][clabel:])
            else:                       #FN
                if FN_option == 'random':
                    # predicting one random class as true that is not true class
                    index = int(np.random.randint(0, C[i], 1))
                    while (index == clabel):
                        index = int(np.random.randint(0, C[i], 1))

                    scores[i][index-1] = acc[i]
                    scores[i][:index-1] = [np.around((1 - acc[i])/(C[i]-1), 3)] * len(scores[i][:index-1])
                    scores[i][index:] = [np.around((1 - acc[i])/(C[i]-1), 3)] * len(scores[i][index:])
                else:
                    # distributing the scores among all
                    scores[i] = [np.around(acc[i]/C[i], 3)] * len(scores[i])
        else:
            threshold = np.random.uniform(0, 1, 1)
            if threshold < acc[i]:    #TN
                scores[i] = [np.around(acc[i]/C[i], 3)] * len(scores[i])     #we multiply by len because we want to have the same number repeated len times
            else:                       #FP
                index = int(np.random.randint(0, C[i], 1)) 
                while (index == clabel):
                    index = int(np.random.randint(0, C[i], 1))

                scores[i][index] = acc[i]
                scores[i][:index] = [np.around((1 - acc[i])/(C[i]-1), 3)] * len(scores[i][:index])
                scores[i][index+1:] = [np.around((1 - acc[i])/(C[i]-1), 3)] * len(scores[i][index+1:])        
    
    return scores

def test_dataset(dataset,config, classifier_type = None):
    """
    takes a calibrated or uncalibrated dataset and checks if the overall accuracy is the
    desired accuracy for all classifiers in classifier_type.
    """

    D = config["D"]
    if classifier_type == 'domain':
        d_acc = config["d_acc"]
        ground_truth = dataset['ground_truth']
        
        for model in range(1, D+1):
            gt = np.array([ground_truth[i][0] == model for i in range(len(ground_truth))])

            scores = dataset['d'+str(model)]
            prediction = np.array([scores[i][0] >= 0.5 for i in range(len(ground_truth))])
            
            acc = np.mean(gt == prediction)
            print('For domain model '+str(model), 'desired accuracy:', d_acc[model-1], 'and dataset accuracy:', acc)

    elif classifier_type == 'intent':
        I = config["I"]
        i_acc = config["i_acc"]
        ground_truth = dataset['ground_truth']
        if_domain = np.array([ground_truth[i][0] for i in range(len(ground_truth))])
        all_ID_acc = []
        all_OOD_acc = []
        for model in range(1, D+1):
            
            # stores the 
            # gt = np.array([ground_truth[i][1] == model for i in range(len(ground_truth))])
            # separate labels that are in-domain and OOD of intent 
            in_domain = ground_truth[if_domain  == model]
            out_domain = ground_truth[if_domain != model]

            in_domain_gt = np.array([in_domain[i][1] for i in in_domain.index])

            # get scores of in and out of domain points
            in_domain_scores = dataset['i'+str(model)][in_domain.index]
            out_domain_scores = dataset['i'+str(model)][out_domain.index]
            
            # make prediction of in-domain points
            prediction = np.array([np.argmax(in_domain_scores[i])+1 for i in in_domain.index])

            # the scores for out-of-domain points should be uniformly distributed 
            out_idx = np.array([np.argmax(out_domain_scores[i])+1 for i in out_domain.index])
            out_domain_pred = np.array([out_domain_scores[j][out_idx[i]-1] ==  out_domain_scores[j][out_idx[i]-2] for i,j in enumerate(out_domain.index)])
            ood_acc = np.mean(out_domain_pred)

            acc = np.mean(in_domain_gt == prediction)
            # print('In_domain labels:', in_domain)
            # print('In domain ground truths of intent:', in_domain_gt)
            # print('In domain scores for intent:', in_domain_scores)
            # print('predictions for in domains:', prediction)
            all_ID_acc.append(acc)
            all_OOD_acc.append(ood_acc)
            print('For intent model '+str(model), 'desired accuracy:', i_acc[model-1], 'and dataset accuracy:', acc, 'and OOD acc:', ood_acc)
        # return all_ID_acc, all_OOD_acc

    elif classifier_type == 'entity':
        E = config["E"]
        e_acc = config["e_acc"]
        ground_truth = dataset['ground_truth']
        
        for model in range(1, D+1):
            
            # stores the 
            # gt = np.array([ground_truth[i][1] == model for i in range(len(ground_truth))])

            # 
            if_domain = np.array([ground_truth[i][0] for i in range(len(ground_truth))])
            in_domain = ground_truth[if_domain  == model]
            out_domain = ground_truth[if_domain != model]

            in_domain_gt = np.array([in_domain[i][2] for i in in_domain.index])

            # get scores of in and out of domain points
            in_domain_scores = dataset['e'+str(model)][in_domain.index]
            out_domain_scores = dataset['e'+str(model)][out_domain.index]
            
            # make prediction of in-domain points
            prediction = np.array([np.argmax(in_domain_scores[i])+1 for i in in_domain.index])

            # the scores for out-of-domain points should be uniformly distributed 
            out_idx = np.array([np.argmax(out_domain_scores[i])+1 for i in out_domain.index])
            out_domain_pred = np.array([out_domain_scores[j][out_idx[i]-1] ==  out_domain_scores[j][out_idx[i]-2] for i,j in enumerate(out_domain.index)])
            ood_acc = np.mean(out_domain_pred)

            acc = np.mean(in_domain_gt == prediction)
            print('For entity model '+str(model), 'desired accuracy:', e_acc[model-1], 'and dataset accuracy:', acc, 'and OOD acc:', ood_acc)

def softmax(x,T=1):
    """
    Calculates the softmax with temperature re-scaling
    """
    if len(x) == 1:
        x_prime = np.log(1 - np.exp(x))
        pot_score = np.exp(x/T)/(np.exp(x/T) + np.exp(x_prime/T))
        x_score = pot_score if pot_score >1e-4 else np.array([1e-4])
        return x_score
        
    return(np.exp(x/T)/np.exp(x/T).sum())

def ECE(x, acc):
    """ Calculates the Expected Calibration Error given accuracy and logits """
    confidence = max(x)
    return abs(confidence - acc)

def overconfident_T_range(acc, dim=1000):
    logit = np.log(np.array([acc]+[(1-acc)/(dim-1) for _ in range(dim-1) ]))
    low = 0.00001; high = 0.999
    mid_top = low + (high - low)/2
    mid_ece = ECE(softmax(logit,mid_top), acc)

    print('ece:',mid_ece, 'mid:',mid_top)
    while mid_ece > 0.11 or mid_ece < 0.098:
        mid_top = low + (high - low)/2
        if mid_ece >0.11:
            low = mid_top
        else:
            high = mid_top
        mid_top = low + (high - low)/2
        mid_ece = ECE(softmax(logit,mid_top), acc)
        print('ece:',mid_ece, 'mid:',mid_top)
    high_ece = mid_ece
    low = 0.00001; high = 0.999
    mid_bot = low + (high - low)/2
    mid_ece = ECE(softmax(logit,mid_bot), acc)

    print('ece:',mid_ece, 'mid:',mid_bot)
    while mid_ece > 0.011 or mid_ece < 0.009:
        mid_bot = low + (high - low)/2
        if mid_ece >0.011:
            low = mid_bot
        else:
            high = mid_bot
        mid_bot = low + (high - low)/2
        mid_ece = ECE(softmax(logit,mid_bot), acc)
        print('ece:',mid_ece, 'mid:',mid_bot)
    print('\n final low ECE:',mid_ece, 'high ECE:', high_ece)
    return mid_top, mid_bot

def underconfident_T_range(acc, dim=1000):
    logit = np.log(np.array([acc]+[(1-acc)/(dim-1) for _ in range(dim-1) ]))
    low = 1.0001; high = 3
    mid_top = low + (high - low)/2
    mid_ece = ECE(softmax(logit,mid_top), acc)

    print('ece:',mid_ece, 'mid:',mid_top)
    while mid_ece > 0.11 or mid_ece < 0.098:
        mid_top = low + (high - low)/2
        if mid_ece >0.11:
            high = mid_top
        else:
            low = mid_top
        mid_top = low + (high - low)/2
        mid_ece = ECE(softmax(logit,mid_top), acc)
        print('ece:',mid_ece, 'mid:',mid_top)
    high_ece = mid_ece
    low = 1.0001; high = 3
    mid_bot = low + (high - low)/2
    mid_ece = ECE(softmax(logit,mid_bot), acc)

    print('ece:',mid_ece, 'mid:',mid_bot)
    while mid_ece > 0.011 or mid_ece < 0.009:
        mid_bot = low + (high - low)/2
        if mid_ece >0.011:
            high = mid_bot
        else:
            low = mid_bot
        mid_bot = low + (high - low)/2
        mid_ece = ECE(softmax(logit,mid_bot), acc)
        print('ece:',mid_ece, 'mid:',mid_bot)
    
    print('\n final low ECE:',mid_ece, 'high ECE:', high_ece)
    return mid_top, mid_bot

def ECE_to_T(ece):
    if ece < 0:
        return 1 + (1.23-1)*(1- (0.1 + ece)/0.1)
    else:
        return 0.01 + (0.6)*(1- (ece)/0.1)
    