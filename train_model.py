import model
import scipy.io as sci
import torch.optim as optim
import torch


def main():
    # Get data
    path = './Datasets/flowers/'
    data = sci.loadmat(path + 'flowers.mat')
    data = torch.from_numpy(data[list(data.keys())[-1]])
    data_rgb = sci.loadmat(path + 'flowers_rgb.mat')
    data_rgb = torch.from_numpy(data_rgb[list(data_rgb.keys())[-1]])

    # Instance model object
    CCNN = model.Model(data, data_rgb, n_endmembers=100)
    # Create optimizer
    optimizer = optim.Adam(CCNN.parameters(), lr=0.001)

    # Define hyperparameters

    # The values of the hyperparameters are the set values from paper
    alpha = 10
    beta = 10
    gamma = 100
    u = 0.001
    v = 0.001
    # Future change: "After a total of 10.000 epochs the lr is reduced to 0"
    num_epochs = 1

    train(CCNN, optimizer, data, data_rgb, alpha, beta, gamma, u, v, num_epochs)


# Create loss loop

def train(model_, optimizer, Z_train, Y_train, alpha, beta, gamma, u, v, num_epochs):

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

        # Print the loss for every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    print("Training finished!")



if __name__ == '__main__':
    main()