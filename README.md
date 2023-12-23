# 2023-UNet-Algorithm

2023-UNet-Algorithm Multichannel U-Net algorithm built in recent internship:) 

The standard U-Net Algorithm in the following video https://www.youtube.com/watch?v=GAYJ81M58y8&t=504s feeds image sizes of 256 by 256 by 1.

Notice that:
-> 1st dimension is image width 
-> 2nd dimension is image height 
-> 3rd dimension is the number of channels.

This isn't really ideal for handling satellite data if we know that the NOAA20 ATMS has 22 channels.
We will be using the same concepts with the decoder and encoder paths.
Also, in Biomedical Image Segmentation, U-Nets are mostly used to solve classification machine learning problems. 
We cannot use classification to predict wind speed and surface pressure of hurricanes.

Changes made: 
-> Softmax to linear activation function 
-> Binary Cross Entropy loss to Mean Squared Error (MSE)

Article has been published in the Remote Sensing Journal! Check it out!
