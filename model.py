import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as sci
import matplotlib.pyplot as plt

# Importing the dataset
indian_pines_path = './Datasets/IndianPines/'
data = sci.loadmat(indian_pines_path + 'Indian_pines_corrected.mat')
data_gt = sci.loadmat(indian_pines_path + 'Indian_pines_gt.mat')
X = data['indian_pines_corrected']
X_gt = data_gt['indian_pines_gt']

#