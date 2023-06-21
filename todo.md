- [x] Create a Github repository 
- [x] Correct all conv layers to have as input the number of channels and the correct output
- [x] Remove Adim  
- [x] Create PSF function 
  - [x] Put MN as input channel and mn as output channel in order to do MN x p become (mn x p)
- [x] Answer the question about the number of endmembers. 
- [x] Review the kernel_size in Econv layer
- [x] Create loss function
- [x] Review the loss function 
- [x] Obtain Hr Multispectral Image (Y) 
- [x] Obtain Lr Hyperspectral Image (Z) 
- [x] Do a forward pass to see the details about the dimensions of the layers
- [x] Create Train loop
- [x] Put linear decay in the learning rate
- [x] Create a prediction.py 
- [x] Save the model after training
- [ ] Create a log file to save the training information
- [ ] Error metrics (RGB RMSE, MRAE, SAM, mSAM, mPSNR, ERGAS)
- [ ] Create hyphotesis about why the training is not working

Some ideas:
- [ ] Remove clamp from the model definition and put it in the training loop (E, PSF and SNF)
- [ ] Put clamp in E, PSF and SNF in the training loop


