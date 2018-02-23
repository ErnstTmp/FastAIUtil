# FastAIUtil
Utilities for the excellent FastAI library (Training NNs in pyTorch)

BestValSave and BestLossSave are classes to use as callbacks in fastai library to save the model with the best validation loss or metrics or the best overall loss during one or more runs of fit() to disk. 

Using the best validation loss or metrics to determine which model to use for inference is a regularization method. 

These classes are especially usefull when optimizing methods for a long time unattended.
