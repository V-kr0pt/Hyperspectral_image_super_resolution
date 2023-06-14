import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

    def __init__(self,Z, Y, h_dim1, h_dim2, h_dim3, n_endmembers, GSD_ratio):
        # call the constructor of the parent class
        super(Model, self).__init__()

        # HSI parameters
        self.HSI_n_rows = Z.shape[0]
        self.HSI_n_cols = Z.shape[1]
        self.n_spectral = Z.shape[2] # number of spectral bands
        self.HSI_n_pixels = self.HSI_n_rows*self.HSI_n_cols

        # MSI parameters
        self.MSI_n_rows = Y.shape[0]
        self.MSI_n_cols = Y.shape[1]
        self.MSI_n_channels = Y.shape[2] # number of color channels
        self.MSI_n_pixels = self.MSI_n_rows*self.MSI_n_cols

        # Number of endmembers
        self.p = n_endmembers

        # lr encoder part
        self.conv1_lr = nn.Conv2d(self.n_spectral, h_dim1, kernel_size=(1,1))  
        self.conv2_lr = nn.Conv2d(h_dim1, h_dim2, kernel_size=(1,1))  
        self.conv3_lr = nn.Conv2d(h_dim2, h_dim3, kernel_size=(1,1))    
        self.conv4_lr = nn.Conv2d(h_dim3, self.p, kernel_size=(1,1))

        # hr encoder part
        self.conv1_hr = nn.Conv2d(self.MSI_n_channels, h_dim1, kernel_size=(1,1))
        self.conv2_hr = nn.Conv2d(h_dim1, h_dim2, kernel_size=(1,1)) 
        self.conv3_hr = nn.Conv2d(h_dim2, h_dim3, kernel_size=(1,1)) 
        self.conv4_hr = nn.Conv2d(h_dim3, self.p, kernel_size=(1,1))

        # SRF function
        self.SRFconv = nn.Conv2d(self.n_spectral, self.MSI_n_channels, kernel_size=(1,1), bias=False) # BIAS FALSE?
        self.SRFnorm = nn.BatchNorm2d(self.MSI_n_channels, affine=False)

        # PSF function
        # kernel size is the ratio of the GSDs between the Z and Y
        # stride is the same as the kernel size
        self.PSFconv = nn.Conv2d(1, 1, kernel_size=GSD_ratio, stride=GSD_ratio, bias=False) # BIAS FALSE?

        # Endmembers Layer 
        self.Econv = nn.Conv2d(self.p, self.n_spectral, kernel_size=(1,1), bias=False)

    def LrHSI_encoder(self, Z):
        h = Z.view((self.n_spectral, self.HSI_n_pixels)) 
        h = nn.LeakyReLU()(self.conv1_lr(h))
        h = nn.LeakyReLU()(self.conv2_lr(h))
        h = nn.LeakyReLU()(self.conv3_lr(h))
        Ah = self.conv4_lr(h) 
        # apply clamp to A to ensure that the abundance values are between 0 and 1
        Ah = torch.clamp(Ah, min=0, max=1)
        return Ah
    
    def HrMSI_encoder(self, Y):
        h = Y.view((self.MSI_n_channels, self.MSI_n_pixels)) 
        h = nn.LeakyReLU()(self.conv1_hr(h))
        h = nn.LeakyReLU()(self.conv2_hr(h))
        h = nn.LeakyReLU()(self.conv3_hr(h))
        A = self.conv4_hr(h) 
        # apply clamp to A
        A = torch.clamp(A, min=0, max=1)
        return A
    
    def endmembers(self, A):
        h = A.view((self.p, -1))
        h = self.Econv(A)
        return h.view((-1, self.n_spectral))           
    
    def SRF(self, x):
        x = x.view((self.n_spectral, -1))
        phi_num = self.SRFconv(x)
        phi = self.SRFnorm(phi_num)        
        return phi.view((self.MSI_n_pixels, self.MSI_n_channels))
    
    def PSF(self, x):
        x = x.view((self.MSI_n_pixels, -1))
        h = self.PSFconv(x)
        return h.view((self.HSI_n_pixels, -1))
    
    def forward(self, Z, Y):
        # applying encoder
        Ah_a = self.LrHSI_encoder(Z) # abundance (mn x p)
        A = self.HrMSI_encoder(Y) # abundance (MN x p)

        # applying PSF
        Ah_b = self.PSF(A) # abundance (mn x p)
        lrMSI_Y = self.PSF(Y) 

        # applying endmembers
        Za = self.endmembers(Ah_a) # lrHSI (mn x n_spectral)
        Zb = self.endmembers(Ah_b) # lrHSI (mn x n_spectral)
        X_ = self.endmembers(A)  # hrHSI (MN x n_spectral)

        # applying SRF
        Y_ = self.SRF(X_) # hrMSI (MN x n_spectral)
        lrMSI_Z = self.SRF(Z)

        return X_, Y_, Za, Zb, A, lrMSI_Z, lrMSI_Y
    
    def loss(self, X, Y, Za, Zb, A, lrMSI_Z, lrMSI_Y, alpha, beta, gamma, delta):
        # loss function
        loss = nn.MSELoss()
        # reconstruction loss
        l1 = loss(X, Y)
        l2 = loss(lrMSI_Z, lrMSI_Y)
        l3 = loss(Za, Zb)
        # regularization loss
        l4 = torch.mean(torch.abs(A))
        l5 = torch.mean(torch.abs(Za))
        l6 = torch.mean(torch.abs(Zb))
        # total loss
        l = l1 + alpha*l2 + beta*l3 + gamma*l4 + delta*l5 + delta*l6
        return l