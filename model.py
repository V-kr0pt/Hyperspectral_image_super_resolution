import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# defining the model class, inheriting from nn.Module
class Model(torch.nn.Module):
    '''
        Model class for the HSI-MSI fusion network
    
        Parameters:
            Z: HSI data
            Y: MSI data
            h_dim1: number of hidden units in the first hidden layer
            h_dim2: number of hidden units in the second hidden layer
            h_dim3: number of hidden units in the third hidden layer
            n_endmembers: number of endmembers
            GSD_ratio: ground sampling distance ratio between LrHSI and HrMSI
    '''

    def __init__(self,Z, Y, n_endmembers=100): # n_endmembers and GSD_ratio defined for Indian Pines dataset
        # call the constructor of the parent class
        super(Model, self).__init__()

        # Number of endmembers
        self.p = n_endmembers

        # HSI parameters
        self.HSI_n_rows = Z.shape[0]
        self.HSI_n_cols = Z.shape[1]
        self.n_spectral = Z.shape[2] # number of spectral bands
        self.HSI_n_pixels = self.HSI_n_rows*self.HSI_n_cols

        # n_spectral -> n_endmembers
        
        h_interval_lr = (n_endmembers - self.n_spectral) // 4
        self.h_dim1_lr_encoder = self.n_spectral + h_interval_lr
        self.h_dim2_lr_encoder = self.h_dim1_lr_encoder + h_interval_lr
        self.h_dim3_lr_encoder = self.h_dim2_lr_encoder + h_interval_lr

        # assuming that all images are squared
        self.GSD_ratio = Y.shape[0] / Z.shape[0]

        assert int(self.GSD_ratio) == self.GSD_ratio, "GSD ratio is not an integer, could you check that?"
        self.GSD_ratio = int(self.GSD_ratio)
        # MSI parameters
        self.MSI_n_rows = Y.shape[0]
        self.MSI_n_cols = Y.shape[1]
        self.MSI_n_channels = Y.shape[2] # number of color channels
        self.MSI_n_pixels = self.MSI_n_rows*self.MSI_n_cols

        # MSI_n_channels -> n_endmembers
        h_interval_hr = (self.MSI_n_channels - self.n_spectral) // 4
        self.h_dim1_hr_encoder = self.n_spectral + h_interval_hr
        self.h_dim2_hr_encoder = self.h_dim1_hr_encoder + h_interval_hr
        self.h_dim3_hr_encoder = self.h_dim2_hr_encoder + h_interval_hr
        
        # lr encoder part 
        self.conv1_lr = nn.Conv2d(self.n_spectral, self.h_dim1_lr_encoder, kernel_size=(1,1))  
        self.conv2_lr = nn.Conv2d(self.h_dim1_lr_encoder, self.h_dim2_lr_encoder, kernel_size=(1,1))  
        self.conv3_lr = nn.Conv2d(self.h_dim2_lr_encoder, self.h_dim3_lr_encoder, kernel_size=(1,1))    
        self.conv4_lr = nn.Conv2d(self.h_dim3_lr_encoder, self.p, kernel_size=(1,1))

        # hr encoder part
        self.conv1_hr = nn.Conv2d(self.MSI_n_channels, self.h_dim1_hr_encoder, kernel_size=(1,1))  
        self.conv2_hr = nn.Conv2d(self.h_dim1_hr_encoder, self.h_dim2_hr_encoder, kernel_size=(1,1))  
        self.conv3_hr = nn.Conv2d(self.h_dim2_hr_encoder, self.h_dim3_hr_encoder, kernel_size=(1,1))    
        self.conv4_hr = nn.Conv2d(self.h_dim3_hr_encoder, self.p, kernel_size=(1,1))

        # SRF function
        self.SRFconv = nn.Conv2d(self.n_spectral, self.MSI_n_channels, kernel_size=(1,1), bias=False) 
        self.SRFnorm = nn.BatchNorm2d(self.MSI_n_channels, affine=False)

        # PSF function
        # kernel size is the ratio of the GSDs between the Z and Y
        # stride is the same as the kernel size
        
        self.PSFconv = nn.Conv2d(1, 1, kernel_size=self.GSD_ratio, stride=self.GSD_ratio, bias=False)

        # Endmembers Layer 
        self.Econv = nn.Conv2d(self.p, self.n_spectral, kernel_size=(1,1), bias=False)

    def LrHSI_encoder(self, Z):
        # we reshape Z to be L x m x n, i.e. (C_in, H_in, W_in)
        h = Z.reshape((self.n_spectral, self.HSI_n_rows, self.HSI_n_cols)).float()
        # we apply the convolutional layers
        h = F.leaky_relu(self.conv1_lr(h))
        h = F.leaky_relu(self.conv2_lr(h))
        h = F.leaky_relu(self.conv3_lr(h))
        Ah = self.conv4_lr(h) 
        # apply clamp to A to ensure that the abundance values are between 0 and 1
        Ah = torch.clamp(Ah, min=0, max=1)
        return Ah
    
    def HrMSI_encoder(self, Y):
        # we reshape Y to be l x M x N, i.e. (C_in, H_in, W_in)
        h = Y.reshape((self.MSI_n_channels, self.MSI_n_rows, self.MSI_n_cols)).float()        
        # we apply the convolutional layers
        h = F.leaky_relu(self.conv1_hr(h))
        h = F.leaky_relu(self.conv2_hr(h))
        h = F.leaky_relu(self.conv3_hr(h))
        A = self.conv4_hr(h) 
        # apply clamp to A
        A = torch.clamp(A, min=0, max=1)
        return A
    
    def endmembers(self, A):
        # we reshape A to be (p x M x N) or (p x m x n) where p is the number of endmembers
        #h = A.view((self.p, -1))
        # apply the conv layer (matrix multiplication) to obtain Z' (mn x L) or X' (MN x L)
        h = self.Econv(A)
        #return h.view((-1, self.n_spectral))           
        return h # Here we expect h to be (p x m x n ) or (p x M x N)
    
    def SRF(self, x):
        # here x will be Z (mn x L) or E (p x L)
        # we'll reshape it to the spectral dimension be the first one
        #x = x.view((self.n_spectral, -1))
        # The conv layer simulates the numerator of SRF function
        phi_num = self.SRFconv(x)
        print(f'phi_num shape: {phi_num.shape}')
        # After normalize, we obtain the spectral degenerated object
        
        # The nn.BatchNorm2d layer expects a 4D input tensor with dimensions (batch_size, num_channels, height, width)
        '''
            By using unsqueeze(0), we add a singleton dimension at index 0 to the x_conv tensor, which represents the batch size. 
            This allows us to provide the expected 4D input to the batch normalization layer.
            SOURCE: GPT, Chat
        '''
        phi_num = phi_num.unsqueeze(0)        
        spectral_degenerated = self.SRFnorm(phi_num) 

        # we remove the singleton dimension (batch_size)
        spectral_degenerated = spectral_degenerated.squeeze(0)

        # apply clamp to ensure that the values are between 0 and 1 
        spectral_degenerated = torch.clamp(spectral_degenerated, min=0, max=1)  
        # we reshape it to the original shape it to (mn x l) if Ylr or (p x l) if Em 
        #return spectral_degenerated.view((-1, self.MSI_n_channels))
        return spectral_degenerated
    
    def PSF(self, x):
        # here x will be A (p x M x N) or Y (l x M x N) 
        #x = x.view((self.MSI_n_pixels, -1))
        # the spatial generated will be Ah (mn x p) or Ylr (mn x l)
        spatial_degenerated = []
        # for each band, p or l
        for band in range(x.shape[0]):
            band_image_tensor = x[band, :, :].reshape((1, x.shape[1], x.shape[2]))
            # the spatial_degenerated cube will be the PSF applied to A or Y 
            # the spatial resolution after do that will be (m x n) due to the 
            # kernel size and stride
            spatial_degenerated.append(self.PSFconv(band_image_tensor).detach().numpy().reshape([x.shape[1], x.shape[2]]))
        return torch.from_numpy(np.array(spatial_degenerated))
    
    def forward(self, Z, Y):
        # applying encoder
        Ah_a = self.LrHSI_encoder(Z) # abundance (p x m x n)
        print(f'Shape Ah_a {Ah_a.shape}')
        A = self.HrMSI_encoder(Y) # abundance (p x M x N)
        print(f'Shape A {A.shape}')

        # applying PSF
        Ah_b = self.PSF(A) # abundance (p x m x n)
        lrMSI_Y = self.PSF(Y.reshape((self.MSI_n_channels, self.MSI_n_rows, self.MSI_n_cols)).float()) 

        # applying endmembers
        Za = self.endmembers(Ah_a) # lrHSI (n_spectral x m x n)
        print(Za.shape)
        Zb = self.endmembers(Ah_b) # lrHSI (n_spectral x m x n)
        print(Zb.shape)
        X_ = self.endmembers(A)  # hrHSI (n_spectral x m x n)
        print(X_.shape)

        # applying SRF
        Y_ = self.SRF(X_) # hrMSI 

        h = Z.reshape((self.n_spectral, self.HSI_n_rows, self.HSI_n_cols)).float()
        lrMSI_Z = self.SRF(h)
        return X_, Y_, Za, Zb, A, Ah_a, Ah_b, lrMSI_Z, lrMSI_Y
    
    
    def loss(self, Z, Y, Za, Zb, Y_, A, Ah_a, Ah_b, lrMSI_Z, lrMSI_Y, alpha, beta, gamma, u, v):
        # loss function
        loss = nn.L1Loss(ord=1)
        # reconstruction loss
        l1 = loss(Za, Z)
        l2 = loss(Zb, Z)
        l3 = loss(Y_, Y)
        l4 = loss(lrMSI_Y, lrMSI_Z)
        Lbase = l1 + alpha*l2 + beta*l3 + gamma*l4
        # Constraint sum2one loss
        l1 = loss(A.sum(dim=1), torch.ones(A.shape[0]))
        l2 = loss(Ah_a.sum(dim=1), torch.ones(Ah_a.shape[0]))
        l3 = loss(Ah_b.sum(dim=1), torch.ones(Ah_b.shape[0]))
        Lsum2one = l1 + l2 + l3
        # Constraint sparsity loss
        a = 1e-4 # sparsity parameter
        Lsparse = 0
        # for each element in A
        for aij in A:
            # Kullback-Leibler divergence
            Lsparse += a*np.log(a/aij) + (1-a)*np.log((1-a)/(1-aij))
        
        # total loss
        l = Lbase + u * Lsum2one + v * Lsparse
        return l