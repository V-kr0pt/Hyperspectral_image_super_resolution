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

    # See values in paper
    alpha = 1000
    beta = 1000
    gamma = 100
    u = 0.001
    v = 0.001
    num_epochs = 10

    train(CCNN, optimizer, data, data_rgb, alpha, beta, gamma, u, v, num_epochs)


# Create loss loop

def train(model_, optimizer, Z_train, Y_train, alpha, beta, gamma, u, v, num_epochs):

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        X_, Y_, Za, Zb, A, Ah_a, Ah_b, lrMSI_Z, lrMSI_Y = model_(Z_train, Y_train).forward()

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