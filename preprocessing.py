import torch.utils.data as data
import torch
import os
import glob
import scipy.io as io
import numpy as np
from scipy import signal

class Dataset(data.Dataset):
    def __init__(self, img, isTrain=True):
        super(Dataset, self).__init__()

        self.img = img
        self.isTrain = isTrain

        (h, w, c) = self.img.shape
        #self.scale_factor = 2 #scale_factor See table 1 page 6 in the article
        #self.sigma = 0.5 #See Section B in the article
        'Ensure that the side length can be divisible'
        #r_h, r_w = h%self.scale_factor, w%self.scale_factor

        'LrHSI'
        self.img_lr = self.generate_LrHSI()
        (self.lrhsi_height, self.lrhsi_width, _) = self.img_lr.shape

        'HrMSI'
        self.img_msi = self.generate_HrMSI()

    def downsamplePSF(self, scale_factor, sigma):
        def matlab_style_gauss2D(shape=(scale_factor,scale_factor), sigma=sigma):
            m,n = [(ss-1.)/2. for ss in shape]
            y,x = np.ogrid[-m:m+1,-n:n+1]
            h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
            h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h
        # generate filter same with fspecial('gaussian') function
        h = matlab_style_gauss2D((scale_factor,scale_factor),sigma)
        if self.img.ndim == 3:
            img_w,img_h,img_c = self.img.shape
        elif self.img.ndim == 2:
            img_c = 1
            img_w,img_h = self.img.shape
            self.img = self.img.reshape((img_w,img_h,1))
        
        out_img = np.zeros((img_w//scale_factor, img_h//scale_factor, img_c))
        for i in range(img_c):
            out = signal.convolve2d(self.img[:,:,i],h,'valid')
            out_img[:,:,i] = out[::scale_factor,::scale_factor]
        return out_img

    def generate_LrHSI(self, scale_factor=2, sigma=0.5):
        img_lr = self.downsamplePSF(scale_factor, sigma)
        return img_lr


    def generate_HrMSI(self):
        srf_bands = {
        'blue': {'center': 0.45, 'fwhm': 0.06},
        'green': {'center': 0.53, 'fwhm': 0.06},
        'red': {'center': 0.66, 'fwhm': 0.03},
        'nir': {'center': 0.86, 'fwhm': 0.03},
        'swir1': {'center': 1.61, 'fwhm': 0.08},
        'swir2': {'center': 2.20, 'fwhm': 0.18}
        }
        num_bands_msi = len(srf_bands)
        msi_data = np.zeros((self.img.shape[0], self.img.shape[1], num_bands_msi))

        for i, band in enumerate(srf_bands):
            center_wavelength = srf_bands[band]['center']
            fwhm = srf_bands[band]['fwhm']

            # Calculate the start and end wavelengths based on the SRF
            start_wavelength = center_wavelength - (fwhm / 2)
            end_wavelength = center_wavelength + (fwhm / 2)
            
            num_channels = 200
            start_wavelength_hsi = 0.4
            end_wavelength_hsi = 2.5

            hsi_wavelengths = np.linspace(start_wavelength_hsi, end_wavelength_hsi, num_channels)

            # Find the corresponding bands in HSI within the specified wavelength range
            start_band = np.argmax(hsi_wavelengths >= start_wavelength)
            end_band = np.argmax(hsi_wavelengths > end_wavelength)

            # Compute the average of HSI bands within the specified range
            msi_data[..., i] = np.mean(self.img[..., start_band:end_band+1], axis=-1)
        return msi_data

    def __len__(self):
        return len(self.imgpath_list)


if __name__ == "__main__":
    indian_pines_path = './Datasets/IndianPines/'
    import scipy.io as sci
    data = sci.loadmat(indian_pines_path + 'Indian_pines_corrected.mat')
    X = data['indian_pines_corrected']
    Images_Generator = Dataset(X)
    HrMSI = Images_Generator.img_msi
    print(HrMSI)
    print(HrMSI[...,0]==HrMSI[...,2])
