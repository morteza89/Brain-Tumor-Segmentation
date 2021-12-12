# Brain-Tumor-Segmentation
A deep learning based algorithm is presented for brain Tumor segmentation in MRI images. 
A dataset of MRI images with their ground truth is available on Kaggle to validate performance of the proposed technique.

In this project Im going to segment Tumor in MRI brain Images with a UNET which is based on Keras. The dataset is available online on Kaggle, and the algorithm provided 99% accuracy with validation loss of 0.11 in just 10 epoches.

Here is the link for accessing to the dataset : 
https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation

And here is my code that I run on kaggle and under just 10 epoches I could achieve accuracy of 99% with just 0.11 validation loss:
https://www.kaggle.com/mortezaheidari/brain-mri-segmentation

The code is so straightforward and easy to use.

## How to use it:
 In the setting parametters part you can change the DataPath to the directory that you saved files from Kaggle.
Then you are able to change number of EPOCHS, BATCH_SIZE, ImgHieght, ImgWidth, Channels from their defult values to train model for other parametters. 
Your model for the best epoch results will be saved in MODEL_SAVE_PATH, so you should change it to a directory you like to save the model. 
If you dont like augmentation, you can turn the "Augmentation" flag to False.
