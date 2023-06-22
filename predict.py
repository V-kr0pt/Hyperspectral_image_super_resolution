import os
import torch
import scipy.io as sci
import matplotlib.pyplot as plt
import model
import preprocessing
from train_model import train 


# Obtaining the high resolution HSI data (X)
path = './Datasets/IndianPines/'
data = sci.loadmat(path + 'Indian_pines_corrected.mat')
hrHSI = data[list(data.keys())[-1]]
# X will be the Tensor of high resolution HSI data 
X = torch.from_numpy(hrHSI.astype(int))

# Creating the low resolution HSI (Z) and high resolution MSI (Y)
Images_Generator = preprocessing.Dataset(hrHSI)
lrHSI = Images_Generator.img_lr
hrMSI = Images_Generator.img_msi

lrHSI = (lrHSI - lrHSI.min()) / (lrHSI.max() - lrHSI.min())
hrMSI = (hrMSI - hrMSI.min()) / (hrMSI.max() - hrMSI.min())

# Transforming the data into Tensors
Z = torch.from_numpy(lrHSI.astype(int))
Y = torch.from_numpy(hrMSI.astype(int))
    
# create the model object
CCNN = model.Model(Z, Y, n_endmembers=100)

# load the model
model_path = './Models/'
model_name = 'model36482.pth'
CCNN.load_state_dict(torch.load(model_path + model_name))

# reshape the data, always the channels first
Z = Z.permute(2, 0, 1) 
Y = Y.permute(2, 0, 1)
Z = (Z - torch.min(Z)) / (torch.max(Z) - torch.min(Z)) 
Y = (Y - torch.min(Y)) / (torch.max(Y) - torch.min(Y))

# Do a forward pass
X_, Y_, Za, Zb, A, Ah_a, Ah_b, lrMSI_Z, lrMSI_Y = CCNN.forward(Z, Y)  

# Visualize the results
fig, ax = plt.subplots(3, 2 , figsize=(25, 25))

ax[0, 0].imshow(X.detach().numpy()[:, :, 1])
ax[0, 0].set_title('High resolution HSI')
ax[0, 1].imshow(X_.detach().numpy()[1, :, :])
ax[0, 1].set_title('Predicted')
ax[1, 0].imshow(Y.detach().numpy()[:, :, 1])
ax[1, 0].set_title('High resolution MSI')
ax[1, 1].imshow(Y_.detach().numpy()[1, :, :])
ax[2, 0].imshow(Z.detach().numpy()[:, :, 1])
ax[2, 0].set_title('Low resolution HSI')
ax[2, 1].imshow(Z.detach().numpy()[:, :, 1] - Za.detach().numpy()[1, :, :])

plt.show()

# Saving image
fig_path = './Results/' 
plt.savefig(fig_path + model_name[:-4] + '.png') 

# Quantitative evaluation
# Calculate the mean spectral angle mapper (mSAM) 
# mSAM = torch.mean(torch.acos(torch.sum(X * X_, dim=0) / (torch.norm(X, dim=0) * torch.norm(X_, dim=0))))
# # Calculate the root mean squared error (RMSE) 
# RMSE = torch.sqrt(torch.mean((X - X_)**2))
# # Calculate the mean PSNR 
# mPSNR = 10 * torch.log10(255**2 / RMSE**2)


