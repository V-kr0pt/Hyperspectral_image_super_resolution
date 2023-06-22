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

# Transforming the data into Tensors
Z = torch.from_numpy(lrHSI.astype(int))
Y = torch.from_numpy(hrMSI.astype(int))
    
# create the model object
CCNN = model.Model(Z, Y, n_endmembers=100)

# load the model
model_path = './Models/'
model_name = 'model2.pth'
CCNN.load_state_dict(torch.load(model_path + model_name))

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
ax[2, 1].imshow(Za.detach().numpy()[1, :, :])

plt.show()

# Quantitative evaluation
# Calculate the mean spectral angle mapper (mSAM) 
# mSAM = torch.mean(torch.acos(torch.sum(X * X_, dim=0) / (torch.norm(X, dim=0) * torch.norm(X_, dim=0))))
# # Calculate the root mean squared error (RMSE) 
# RMSE = torch.sqrt(torch.mean((X - X_)**2))
# # Calculate the mean PSNR 
# mPSNR = 10 * torch.log10(255**2 / RMSE**2)


