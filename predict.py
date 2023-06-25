import os
import torch
import scipy.io as sci
import matplotlib.pyplot as plt
import model
import preprocessing
import numpy as np

def main(model_path='./Models/article_model.pth', plot=True, save_figure=False):
    # Obtaining model name
    if model_path.split('/')[1] == 'Grid_Search':
        model_name = model_path.split('/')[-2] + '_' + model_path.split('/')[-1] 
    else:
        model_name = model_path.split('/')[-1]

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

    CCNN.load_state_dict(torch.load(model_path))
    # permuting the data to have the channels first
    X = X.permute(2, 0, 1)
    Z = Z.permute(2, 0, 1) 
    Y = Y.permute(2, 0, 1)

    # normalizing the data
    Z = (Z - torch.min(Z)) / (torch.max(Z) - torch.min(Z)) 
    Y = (Y - torch.min(Y)) / (torch.max(Y) - torch.min(Y))


    # Do a forward pass
    X_, Y_, Za, Zb, A, Ah_a, Ah_b, lrMSI_Z, lrMSI_Y = CCNN.forward(Z, Y)  

    # Visualize the results
    X, X_ = normalize(X, X_)
    
    
    fig, ax = plt.subplots(3, 2)
    #plt.set_supertitle('Results of the CCNN model')
    plt.subplots_adjust(left=0.12, bottom=0.048, right=0.9, top=0.93, wspace=0.45, hspace=0.524)

    ax[0, 0].imshow(X.detach().numpy()[1, :, :])
    ax[0, 0].set_title('hrHSI')
    ax[0, 1].imshow(X_.detach().numpy()[1, :, :])
    ax[0, 1].set_title('Predicted hrHSI')

    ax[1, 0].imshow(Y.detach().numpy()[1, :, :])
    ax[1, 0].set_title('hrMSI')
    ax[1, 1].imshow(Y_.detach().numpy()[1, :, :])
    ax[1, 1].set_title('Predicted hrMSI')

    ax[2, 0].imshow(Z.detach().numpy()[1, :, :])
    ax[2, 0].set_title('lrHSI')
    ax[2, 1].imshow(Za.detach().numpy()[1, :, :])
    ax[2, 1].set_title('Predicted lrHSI')

    fig.colorbar(ax[0, 0].imshow(X.detach().numpy()[1, :, :]), ax=ax[0, 0])
    fig.colorbar(ax[0, 1].imshow(X_.detach().numpy()[1, :, :]), ax=ax[0, 1])
    fig.colorbar(ax[1, 0].imshow(Y.detach().numpy()[1, :, :]), ax=ax[1, 0])
    fig.colorbar(ax[1, 1].imshow(Y_.detach().numpy()[1, :, :]), ax=ax[1, 1])
    fig.colorbar(ax[2, 0].imshow(Z.detach().numpy()[1, :, :]), ax=ax[2, 0])
    fig.colorbar(ax[2, 1].imshow(Za.detach().numpy()[1, :, :]), ax=ax[2, 1])

    if save_figure:
        if not os.path.exists('./Results'):
            os.mkdir('./Results')
        #save the figure as modelname.png 
        plt.savefig('./Results/' + model_name[:-4] +'.png')
        print("Figure saved!")
    
    if plot:   
        # showing the figure
        plt.show()
    
    

    # Quantitative evaluation
    # Calculate the mean spectral angle mapper (mSAM) 
    
    #mSAM = torch.mean(torch.acos(torch.sum(X * X_, dim=0) / (torch.norm(X, dim=0) * torch.norm(X_, dim=0))))
    # # Calculate the root mean squared error (RMSE) 
    RMSE = torch.sqrt(torch.mean((X - X_)**2))
    # # Calculate the mean PSNR 
    mPSNR = 10 * torch.log10(255**2 / RMSE**2)
    return RMSE.item()

def normalize(X, X_):
    X = (X - torch.min(X)) / (torch.max(X) - torch.min(X))
    X_ = (X_ - torch.min(X_)) / (torch.max(X_) - torch.min(X_))
    return X, X_

if __name__ == '__main__':
    main()
    