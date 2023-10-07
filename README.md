# Unified-NLU-Reranker
 A unified Natural Language Understanding reranker with deep reinforcement learning

## Installing
Install the requirements using:

``
conda create --name <env> --file requirements.txt
``

## Generating dataset
To generate the dataset, create a config file similar to ``dataset.yaml`` and run ``generate_dataset.py`` using the file name.

## Model development
To start onboarding a new model, follow the following:
1. Write the agent in ``agents.py`` with a predict function that takes in a pytorch tensor of shape $batch \times 3H$ and outputs a vector $batch \times 1$ with class predictions, where $H$ is the total number of hypotheses.
2. In ``dev_test.py``, initialize your agent, and its optimizer (for ex. ``torch.optim.SGD``) and criterion (for ex. ``torch.nn.CrossEntropyLoss``). If your agent does not use any of the latter two, see point number 5.
3. In ``configs/`` directory, add a new file using the format of ``AGENT_NAME.yaml`` that requires description of the dataset you are using and the agent parameters.
4. To ensure the dataset works first, add it to ``datasets`` folder and update the path in the config file. If it works correctly, then you should run the function ``all_sklearn_tests`` in ``dev_test.py`` before running your agent. It will run several different classifiers and report each one's training and testing accuracies, as well as the test SemER values. Take these with a grain of salt, however, since no hyper-paramter tuning is done on these. For simpler datasets, this should not be a problem but as of May 12 debugging is still left.
5. In case your agent cannot use an optimizer and criterion like standard PyTorch ``nn.module`` agents, you can go to ``utils/train_util.py`` and in the training and validating one epoch functions add a new condition for training with whatever algorithm is required for your agent. Take a look at the implementation of LinUCB classifier for reference.

## Acknowledgments
The codebase was written by [Yugantar Prakash](https://www.yugantar.me/), [Shatabdi Bhise](https://www.linkedin.com/in/shatabdibhise/), [Alexandra Burushkina](https://www.linkedin.com/in/alexandra-burushkina-5417a7215/) and [Aditya Kuppa](https://www.linkedin.com/in/adityakuppa/) for our project in COMPSCI 696DS. We would like to thank [Yuguang Yue](https://www.linkedin.com/in/yuguang-yue-95aba08a/) and [Rico Angell](https://people.cs.umass.edu/~rangell/) for their invaluable insights.

<!-- git push --set-upstream origin HEAD:main-dev-tests -->