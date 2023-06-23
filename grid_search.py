import os
import model
from datetime import datetime 
import preprocessing
import scipy.io as sci
import torch
import torch.optim as optim
import numpy as np
import sys
import itertools
import predict

def main():
    num_epochs = 5
    # Ask the user for the model name
    try:
        model_name = sys.argv[1]
        print(sys.argv)
    except:
        current_datetime = datetime.now()
        year = current_datetime.year
        month = current_datetime.month
        day = current_datetime.day
        hour = current_datetime.hour
        minute = current_datetime.minute
        model_name = f'model_{year}-{month}-{day}-{hour}-{minute}' 
    folder_base_name = 'Grid_Search/Grid_Search_'   
    os.makedirs(folder_base_name + model_name)
    model_dir = folder_base_name + model_name
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
    
    # lrHSI = normalize(lrHSI)
    # hrMSI = normalize(hrMSI)
    
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
    
    alpha_values = np.logspace(-3, 3, 3)
    beta_values = np.logspace(-3, 3, 3)
    gamma_values = np.logspace(-3, 3, 3)
    u_values = np.logspace(-3, 3, 3)
    v_values = np.logspace(-3, 3, 3)
    combinations = list(itertools.product(alpha_values, beta_values, gamma_values, u_values, v_values))
    # Future change: "After a total of 10.000 epochs the lr is reduced to 0"
    
    metadata_file = open(model_dir + '/metadata.txt', "x")
    metadata_file.write('index,alpha,beta,gamma,u,v,RMSE\n')
    for index, combination in enumerate(combinations):
        # Create optimizer
        CCNN = model.Model(Z, Y, n_endmembers=100)
        optimizer = optim.Adam(CCNN.parameters(), 
                           betas = (0.9, 0.999),
                           eps = 1e-08,
                           lr=0.01)
        alpha = combination[0]
        beta = combination[1]
        gamma = combination[2]
        u = combination[3]
        v = combination[4]
        file_name = model_dir + '/model_' + str(index) + '.pth'
        metadata_file.write(str(index) + ',' + str(alpha) + ',' + str(beta) + ',' + str(gamma) + ',' + str(u) + ',' + str(v))
        last_loss = train(CCNN, optimizer, Z, Y, alpha, beta, gamma, u, v, num_epochs, file_name)
        RMSE = predict.main(model_name=file_name, plot=False)
        metadata_file.write(',' + str(RMSE) + '\n')
    metadata_file.close()
def normalize(input, axis=2):
    sum = np.sum(input, axis=axis, keepdims=True)
    output = input / sum
    return output


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
        
        with torch.no_grad():
            for p in model_.Econv.parameters():
                p.clamp_(0, 1)
            for p in model_.PSFconv.parameters():
                p.clamp_(0, 1)
            for p in model_.SRFconv.parameters():
                p.clamp_(0, 1)
                
        
        scheduler.step()

        # Print the loss for every epoch
        last_loss = loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {last_loss}, last_lr: {scheduler.get_last_lr()}")
        
    print("Training finished!")

    # creating folder to save the model 
    #path_model = './Models/'
    # if Model_checkpoint folder does not exist, create it
    #if not os.path.exists(path_model):
        #os.makedirs(path_model)
    # saving the model
    
    torch.save(model_.state_dict(), model_name_)
    print(model_name_ + " saved!")
    return last_loss

if __name__ == '__main__':
    main()