import os
import model
import preprocessing
import scipy.io as sci
import torch
import torch.optim as optim


def main():
    # Ask the user for the model name
    model_name = input("Input the model name: ")
    if model_name == '':
        model_name = 'model' 
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
    num_epochs = 1

    train(CCNN, optimizer, Z, Y, alpha, beta, gamma, u, v, num_epochs, model_name)


# Create loss loop

def train(model_, optimizer, Z_train, Y_train, alpha, beta, gamma, u, v, num_epochs, model_name='model.pth'):
    
    # Create scheduler to implement the learning rate decay
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=.5, end_factor=0, total_iters=10000)
    
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

if __name__ == '__main__':
    main()