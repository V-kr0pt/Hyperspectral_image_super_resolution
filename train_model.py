import os
import model
from datetime import datetime 
import preprocessing
import scipy.io as sci
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def main(plot=True, save_figure=True):
    # Ask the user for the model name
    model_name = input("Input the model name: ")
    if model_name == '':
        current_datetime = datetime.now()
        year = current_datetime.year
        month = current_datetime.month
        day = current_datetime.day
        hour = current_datetime.hour
        minute = current_datetime.minute
        model_name = f'model_{year}-{month}-{day}-{hour}-{minute}' 
    model_name = model_name + '.pth'

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
    
    # Instance model object
    CCNN = model.Model(Z, Y, n_endmembers=100)
    # Create optimizer
    optimizer = optim.Adam(CCNN.parameters(), 
                           betas = (0.9, 0.999),
                           eps = 1e-08,
                           lr=0.01)

    # Define hyperparameters

    # The values of the hyperparameters are the set values from paper
    alpha = 10
    beta = 10
    gamma = 100
    u = 0.001
    v = 0.001
    # Future change: "After a total of 10.000 epochs the lr is reduced to 0"
    num_epochs = 2000

    train(CCNN, optimizer, Z, Y, alpha, beta, gamma, u, v, num_epochs, model_name, plot=plot, save_figure=save_figure)


# Create loss loop

def train(model_, optimizer, Z_train, Y_train, alpha, beta, gamma, u, v, num_epochs, model_name='model.pth', plot=False, save_figure=False):
    
    # Create scheduler to implement the learning rate decay
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=.5, end_factor=0, total_iters=10000)
    
    # reshape the data, always the channels first
    Z_train = Z_train.permute(2, 0, 1) 
    Y_train = Y_train.permute(2, 0, 1)

    #normalize the data
    Z_train = (Z_train - torch.min(Z_train)) / (torch.max(Z_train) - torch.min(Z_train)) 
    Y_train = (Y_train - torch.min(Y_train)) / (torch.max(Y_train) - torch.min(Y_train))
    
    losses = []
    # Training loop 
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear gradients
        # Forward pass
        X_, Y_, Za, Zb, A, Ah_a, Ah_b, lrMSI_Z, lrMSI_Y = model_.forward(Z_train, Y_train)  

        # Compute the loss
        loss = model_.loss(Z_train, Y_train, Za, Zb, Y_, A, Ah_a, Ah_b, lrMSI_Z, lrMSI_Y, alpha, beta, gamma, u, v)

        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Print the loss for every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, last_lr: {scheduler.get_last_lr()}")

    print("Training finished!")

    # creating folder to save the model 
    path_model = './Models/'
    # if Model_checkpoint folder does not exist, create it
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    # saving the model
    torch.save(model_.state_dict(), path_model + model_name)
    print(model_name + " saved!")
    
    # saving the training losses 
    losses = np.array(losses) 

    path_train_history = './Train_History/' 
    path_numpy_train_history = path_train_history + 'Losses/'     
    path_fig_train_history = path_train_history + 'Figures/'

    if not os.path.exists(path_numpy_train_history):
        os.makedirs(path_numpy_train_history)    

    np.save(path_train_history + model_name[:-4], losses) 

    plt.plot(losses)
    plt.grid(True)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    if save_figure:
        if not os.path.exists(path_fig_train_history):
            os.makedirs(path_fig_train_history)
        plt.savefig(path_fig_train_history + model_name[:-4] + '.png')
        print("Figure saved!")
    
    if plot:    
        plt.show()
    

if __name__ == '__main__':
    main()