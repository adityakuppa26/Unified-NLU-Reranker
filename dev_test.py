"""
This script achieves two things:
    - unit tests: provide the basic obstacles that a model should pass
    - visualizations: provide visualizations and metrics on how the model performs
                      on unit tests

This script allows for a more efficient developmental phase for the models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:',device)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
np.set_printoptions(suppress=True)

from utils.util import AnnotatedDataset, ToTensor, labels_to_hypothesis, semER, interER, random_split, load_config
import agents
from utils.train_util import train_one_epoch, valid_one_epoch
# from agents import LinearClassifier, LinUCB, nonLinearClassifier, DistributedLinearClassifier
# from train import train_one_epoch, valid_one_epoch

SCORE_DATASET: '/Users/shatabdibhise/Documents/Semester-IV/696DS/Unified-NLU-Reranker/new_scores_dataset.csv'
INDEX_DATASET: '/Users/shatabdibhise/Documents/Semester-IV/696DS/Unified-NLU-Reranker/new_idx_dataset.csv'

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# PCA and t-SNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import time


def all_sklearn_tests(X,Y, dataset):

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        #"Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    # preprocess dataset, split into training and test part
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.4, random_state=42
    )

    # iterate over classifiers
    for name, clf in zip(names, classifiers):

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)

        # accuracies after training
        train_score = clf.score(X_train, y_train)
        score = clf.score(X_test, y_test)

        # validation for SemER
        outputs = clf.predict(X_test)
        pred_hyp = labels_to_hypothesis(outputs, dataset)

        # ground truth hypotheses
        gt_labels = y_test
        gt_hyp = labels_to_hypothesis(gt_labels, dataset)

        #performance measures
        sem_error = semER(pred_hyp, gt_hyp)
        inter_error = interER(pred_hyp, gt_hyp)

        print(name, 'accuracy on test:', train_score, 'accuracy on train:', score,
                    'with validation SemER:', sem_error)

def svm_testing(X, Y):
    clf = svm.NuSVC(gamma = "auto")
    clf.fit(X,Y)

    predicted = clf.predict(X)

    print('Acc for RBF kernel:',accuracy_score(Y, predicted))


    clf = svm.NuSVC(kernel = 'linear',gamma = "auto")
    clf.fit(X,Y)

    predicted = clf.predict(X)

    print('Acc for linear kernel:',accuracy_score(Y, predicted))


    clf = svm.NuSVC(kernel = 'sigmoid',gamma = "auto")
    clf.fit(X,Y)

    predicted = clf.predict(X)

    print('Acc for sigmoid kernel:',accuracy_score(Y, predicted))

def PCA_visualization(data_X, data_Y, total_classes, viz_components = 3, n_components = 10):
    """
    PCA visulization. 
        data_X: array of shape (batch, feature_dims)
        data_Y: array of shape (batch, )
        total_classes: total number of hypotheses
        viz_components: total components to visualize of PCA
        n_components: components to do PCA
    """
    pca = PCA(n_components = n_components)
    pca_result = pca.fit_transform(X = data_X)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    df = {}
    for i in range(viz_components):
        df['pca-'+str(i+1)] = pca_result[:,i]
    df = pd.DataFrame(data = df)
    df['y'] = data_Y
    
    if viz_components == 2:

        plt.figure(figsize = (16,10))
        sns.scatterplot(
            x = 'pca-'+str(1), y = 'pca-'+str(2),
            hue = "y",
            palette = sns.color_palette("hls", total_classes),
            data = df,
            legend = "full",
            alpha = 0.3
        )
        plt.savefig('plots/PCA_2_components.png')

    elif viz_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            xs=df['pca-'+str(1)], 
            ys=df['pca-'+str(2)], 
            zs=df['pca-'+str(3)], 
            c=df["y"], 
            cmap='tab10'
        )
        fig.savefig('plots/PCA_3_components.png')

def tSNE_visualization(data_X, data_Y, total_classes, viz_components = 3,n_components=10):

    pca = PCA(n_components = n_components)
    pca_result = pca.fit_transform(X = data_X)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    df = {}
    for i in range(viz_components):
        df['pca-'+str(i+1)] = pca_result[:,i]
    df = pd.DataFrame(data = df)
    df['y'] = data_Y
    
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tnse_results = tsne.fit_transform(data_X)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    df['tsne-2d-one'] = tnse_results[:,0]
    df['tsne-2d-two'] = tnse_results[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", total_classes),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.savefig('plots/tSNE.png')


def train_agent(dataset, agent, optimizer=None, criterion=None, agent_type="linear", framework="unif", config = None):
    """ Main Training Loop """
    
    # create train and validation data loaders
    # dataset = AnnotatedDataset(score_csv = config["SCORE_DATASET"],
    #                              idx_csv = config["INDEX_DATASET"],
    #                              transform = ToTensor())
    train_data, valid_data, test_data = random_split(dataset, config["train_valid_test"])
    train_loader = DataLoader(train_data, batch_size = config["batch_size"],
                            shuffle = True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size = config["batch_size"],
                            shuffle = True, num_workers=0)
    I = config["I"]
    E = config["E"]
    Hspaces = [0]+[I[i]*E[i] for i in range(len(I))]
    config["Hspaces"] = Hspaces
    # send model to device
    agent.to(device)

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
        
        if epoch%10==0:
            print('Epoch:', epoch,'Training loss:', train_loss, 'Valid SemER:', valid_sem_err)


def main():
    # folder to load config file
    CONFIG_PATH = "configs/"
    # get config from YAML
    config = load_config("AGENT_NAME.yaml", CONFIG_PATH)

    # dataset tests first
    SCORE_DATASET = config["SCORE_DATASET"]
    INDEX_DATASET = config["INDEX_DATASET"]
    dataset = AnnotatedDataset(score_csv = SCORE_DATASET,
                                 idx_csv = INDEX_DATASET,
                                 transform = ToTensor())

    X,Y = dataset.get_X_Y()
    print('X:', X.shape, 'Y:', Y.shape)
    print('scores:',X[0])
    print('reshaped with (-1,)',X.reshape(-1, 12*3)[0])
    print('reshaped with (,-1)',X.reshape((len(dataset), -1))[0])
    print('Y:',Y[0])
    Y = Y.reshape((-1,))
    # svm_testing(X,Y)
    # all_sklearn_tests(X,Y, dataset)

    # PCA_visualization(data_X=X, data_Y=Y, total_classes=dataset.Hall, viz_components = 3)
    # tSNE_visualization(data_X=X, data_Y=Y, total_classes=dataset.Hall, viz_components = 3)
    
    # print(config)
    # Initialize agent
    Hall = config["Hall"]
    num_of_classifiers = config["num_of_classifiers"]
    agent = agents.LinearClassifier(input_dim = Hall*num_of_classifiers,
                                    num_of_classes = Hall)
    optimizer = optim.SGD(agent.parameters(),
                            lr = config["lr"],
                            momentum = config["momentum"])
    criterion = nn.CrossEntropyLoss()

    # train
    train_agent(dataset = dataset,
                agent=agent,
                optimizer= optimizer,
                criterion = criterion,
                agent_type=config["agent_type"],
                framework=config["framework"],
                config = config)

if __name__=="__main__":
    main()