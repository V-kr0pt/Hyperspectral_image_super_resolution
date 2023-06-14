- [x] The dimension of Convolution Layers:
  - I am not sure about the dimensions of conv nets.  
  
  - [x] First about, the images. We have images "Z" that are m x n x L, what we should do to apply the convolution layer? 

  Hyphotesis: 

    We transform the image in a  2D vector, (mn x L) with it'll apply the convolution layer like: Conv2D(mn, hidden_size, (1,1)), why? I don't know.
  
  Answer:
      The input of a convolution layer is the number of channels, it means the number of bands for HSI image.

  - [x] Ok, after, how about the endmembers conv layer? We can see in the article that it'll receive as entry the "A" matrix, that will be if Ah (mn x p) or if A (MN x p), and the output will be or Za (mn x L) or X (MN x L).
  
    ![Image from the article, referent to the previous question](image.png)


  Hyphotesis: 

    The endmember conv layer have to transform the last dimension "p", it's the number of endmembers, into the number of bands "L". 


Maybe this sentence from the article can help us. 

    "The dimension of the endmembers conv layer is px1x1xL where px1x1 is the kernel and L the number os kernels. "p" is the number of endmembers. For the case of indian pines dataset, p = 16."

  - [x] How about PSF layer?

        "The PSF, a convolution layer with 1 input channel and 1 output channel is implemented for every band of the abundance A."  

---

- [x] The number of endmemembers it's something known by the user or we have to find it? If the second option, how?

      the number of endmembers will be an hyperparameter. In pag 9, the article says that "Therefore, a larger number of endmembers allows the model to be more representative. Although the number of endmembers is assumed to be equal to the number of pure spectral bases in the linear unmixing, the number of endmembers can also be larger than the actual number of pure bases because the convolution weight matrix E can contain mixed material"

- [ ] How to guarantee that A will have MN x p dimension, since the conv layer without any padding will reduce the dimension of the input?
    
- [ ] PSF and SRF conv layers have the bias set to off?

- [ ] The article removed more bands than the correct Indian Pines dataset. What dataset we should use?

- [ ] The article says in the subsection E.number of endmembers that "p" represents the kernel size of the shared convolutional layer. But in the subsection A. Coupled Autoencoder Network for Image Fusion, it says that all the convolutional kernel sizes are set to be 1x1, except for the PSF conv layer. This information also appears in the Figure 2, where the shared conv layer has as name 1x1Conv.  

- [x] How to obtain the HrMSI data?
 
      The dataset that we have is already with HrHSI images, so we apply SRF to obtain HrMSI images. "To simulate the HrMSI, the SRF for the blue to SWIR2 bands of the Landsat 8 were used" from Implementation Details pg. 6
    

