import os
import model
from datetime import datetime 
import preprocessing
import scipy.io as sci
import torch
import torch.optim as optim
import numpy as np
import itertools
import predict
import pandas as pd
import argparse

def main():
    # I think 7 is the ideal number of points for the logspaces, but it's too costly
    alpha_values = np.logspace(-3, 3, 3)
    beta_values = np.logspace(-3, 3, 3)
    gamma_values = np.logspace(-3, 3, 3)
    u_values = np.logspace(-3, 3, 3)
    v_values = np.logspace(-3, 3, 3)
    num_epochs = 100
    '''
    Usage:
    Change the values of the desired hyperparameters on this file before grid searching
    - To grid search: 
    python grid_search.py <model_name>
    - To visualize the best hyperparameters based on the RMSE for a given model:
    python grid_search <model_name> --display
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Name of the file to open')
    parser.add_argument('--display', action='store_true', help='Display the models sorted by the RMSE')
    args = parser.parse_args()
    folder_base_name = 'Grid_Search/'  

    # Ask the user for the model name
    if (args.filename):
        model_name = args.filename
        model_dir = folder_base_name + model_name
        try:
            if args.display:
                dataframe = pd.read_csv(model_dir + '/metadata.txt')
                dataframe = dataframe.sort_values('RMSE')
                print(dataframe)
                return
        except:
            print("There is no model to display yet. Do the grid search and after try to display the results")
    else:
        current_datetime = datetime.now()
        year = current_datetime.year
        month = current_datetime.month
        day = current_datetime.day
        hour = current_datetime.hour
        minute = current_datetime.minute
        model_name = f'model_{year}-{month}-{day}-{hour}-{minute}' 
        model_dir = folder_base_name + model_name    

    os.makedirs(folder_base_name + model_name)
    
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
  
    # normalize lrHSI and hrMSI
    lrHSI = (lrHSI - lrHSI.min()) / (lrHSI.max() - lrHSI.min())
    hrMSI = (hrMSI - hrMSI.min()) / (hrMSI.max() - hrMSI.min())
    
    # Transforming the data into Tensors
    Z = torch.from_numpy(lrHSI.astype(int))
    Y = torch.from_numpy(hrMSI.astype(int))  

    combinations = list(itertools.product(alpha_values, beta_values, gamma_values, u_values, v_values))
    
    metadata_file = open(model_dir + '/metadata.txt', "x")
    metadata_file.write('index,alpha,beta,gamma,u,v,RMSE\n')
    for index, combination in enumerate(combinations):
        # Create model
        CCNN = model.Model(Z, Y, n_endmembers=100)
        # Create optimizer
        optimizer = optim.Adam(CCNN.parameters(), 
                           betas = (0.9, 0.999),
                           eps = 1e-08,
                           lr=0.01)
        # Gridsearch hyperparameters values 
        alpha = combination[0]
        beta = combination[1]
        gamma = combination[2]
        u = combination[3]
        v = combination[4]
        # Saving each model and respective hyperparameters in the metadata_file 
        file_name = model_dir + '/model_' + str(index) + '.pth'
        metadata_file.write(str(index) + ',' + str(alpha) + ',' + str(beta) + ',' + str(gamma) + ',' + str(u) + ',' + str(v))
        
        # obtaining the last loss
        last_loss = train(CCNN, optimizer, Z, Y, alpha, beta, gamma, u, v, num_epochs, file_name)
        
        # obtaining RMSE from prediction
        RMSE = predict.main(model_name=file_name, plot=False)

        # also saving this value in the metadata
        metadata_file.write(',' + str(RMSE) + '\n')
    metadata_file.close()
    
    # printing the list of best models obtained
    dataframe = pd.read_csv(model_dir + '/metadata.txt')
    dataframe = dataframe.sort_values('RMSE')
    print(f"the best 5 models: \n {dataframe.head(5)}")
    print(f"the worst 5 models: \n {dataframe.tail(5)}")

# Create loss loop
def train(model_, optimizer, Z_train, Y_train, alpha, beta, gamma, u, v, num_epochs, model_name_='model.pth'):
    print(model_name_)
    # Create scheduler to implement the learning rate decay
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=.5, end_factor=0, total_iters=num_epochs)
    
    # reshape the data, always the channels first
    Z_train = Z_train.permute(2, 0, 1) 
    Y_train = Y_train.permute(2, 0, 1)

    # Training loop 
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear gradients
        # Forward pass
        X_, Y_, Za, Zb, A, Ah_a, Ah_b, lrMSI_Z, lrMSI_Y = model_.forward(Z_train, Y_train)  

        # Compute the loss
        loss = model_.loss(Z_train, Y_train, Za, Zb, Y_, A, Ah_a, Ah_b, lrMSI_Z, lrMSI_Y, alpha, beta, gamma, u, v)

        # Backward pass
        loss.backward()
        optimizer.step()             
        scheduler.step()

        # Print the loss for every epoch
        last_loss = loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {last_loss}, last_lr: {scheduler.get_last_lr()}")
        
    print("Training finished!")
    
    # Saving the trained model
    torch.save(model_.state_dict(), model_name_)
    print(model_name_ + " saved!")
    return last_loss

if __name__ == '__main__':
    main()