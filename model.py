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
            N_endmembers: number of endmembers
    '''
    def __init__(self,Z, Y, h_dim1, h_dim2, h_dim3, n_endmembers):
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
        self.conv1_lr = nn.Conv2d(self.HSI_n_pixels, h_dim1, kernel_size=(1,1))  
        self.conv2_lr = nn.Conv2d(h_dim1, h_dim2, kernel_size=(1,1))  
        self.conv3_lr = nn.Conv2d(h_dim2, h_dim3, kernel_size=(1,1))    
        self.conv4_lr = nn.Conv2d(h_dim3, self.A_dim, kernel_size=(1,1))

        # hr encoder part
        self.conv1_hr = nn.Conv2d(self.MSI_n_pixels, h_dim1, kernel_size=(1,1))
        self.conv2_hr = nn.Conv2d(h_dim1, h_dim2, kernel_size=(1,1)) 
        self.conv3_hr = nn.Conv2d(h_dim2, h_dim3, kernel_size=(1,1)) 
        self.conv4_hr = nn.Conv2d(h_dim3, self.A_dim, kernel_size=(1,1))

        # SRF function
        self.SRFconv = nn.Conv2d(self.n_spectral, self.MSI_n_pixels, kernel_size=(1,1), bias=False) # BIAS FALSE?
        self.SRFnorm = nn.BatchNorm2d(self.MSI_n_pixels, affine=False)

        # PSF function
        self.PSFconv = nn.Conv2d(1, 1, kernel_size=(3,3), padding=1, bias=False) # BIAS FALSE?

        # Endmembers Layer 
        self.Econv = nn.Conv2d(self.p, self.n_spectral, kernel_size=(1,1), bias=False)

    def LrHSI_encoder(self, Z):
        h = Z.view((-1, self.HSI_n_pixels)) 
        h = nn.LeakyReLU()(self.conv1_lr(h))
        h = nn.LeakyReLU()(self.conv2_lr(h))
        h = nn.LeakyReLU()(self.conv3_lr(h))
        Ah = self.conv4_lr(h) 
        # apply clamp to A to ensure that the abundance values are between 0 and 1
        Ah = torch.clamp(Ah, min=0, max=1)
        return Ah
    
    def HrMSI_encoder(self, Y):
        h = Y.view((-1, self.MSI_n_pixels)) 
        h = nn.LeakyReLU()(self.conv1_hr(h))
        h = nn.LeakyReLU()(self.conv2_hr(h))
        h = nn.LeakyReLU()(self.conv3_hr(h))
        A = self.conv4_hr(h) 
        # apply clamp to A
        A = torch.clamp(A, min=0, max=1)
        return A
    
    def endmembers(self, A):
        h = A.view((-1, self.p))
        h = self.Econv(A)
        return h.view((-1, self.n_spectral))    
        
    
    def SRF(self, x):
        x = x.view((-1, self.n_spectral, 1, 1))
        phi_num = self.SRFconv(x)
        msi_img = self.SRFnorm(phi_num)        
        return msi_img.view((self.MSI_n_pixels, self.MSI_n_channels))
    
    def PSF(self, x):
        pass
    
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