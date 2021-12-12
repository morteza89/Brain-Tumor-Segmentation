
# ------------------------------------------------------------------------------
# Name:        Image Segmentation with U-Net
# Purpose:	   MRI brain Tumor segmentation .
#
# Author:      Morteza Heidari
#
# Created:     12/12/2021
# ------------------------------------------------------------------------------
# In this code we will use brain MR together with the manual FLAIR abnormality segmentation masks to segment the tumor.
# taking advantage of CNN with UNET as the segmentation model.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model, save_model
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
######
## Setting parametters
DataPath = "/kaggle/input/lgg-mri-segmentation/kaggle_3m/"  # Path to the data
EPOCHS = 35
BATCH_SIZE = 32
ImgHieght = 256
ImgWidth = 256
Channels = 3
MODEL_SAVE_PATH = "./models/model_unet_3m.h5"  # directory to save the  best model
Augmentation = True  # True or False defines whether to use data augmentation or not
BachNopm_Dropout = True #  it is better if we have batch normalization, we dont use dropout, and vise versa.
####


class Data_PreProcessing():
    def __init__(self):
        self.DataPath = DataPath

    def prepare_data(self):
        dirs = []
        images = []
        masks = []
        for dir, subdirs, files in os.walk(self.DataPath):
            for file in files:
                if 'mask' in file:
                    dirs.append(dir.replace(self.DataPath, ''))
                    masks.append(file)
                    images.append(file.replace('_mask', ''))
        # print size of three lists to check if they are equal
        print('Number of directories:', len(dirs), len(images), len(masks))
        # now create a dataframe with the three lists
        image_df = pd.DataFrame({'dir': dirs, 'image': images, 'mask': masks})
        # configuring the dataframe with the whole path to the images and masks
        image_df['image_path'] = self.DataPath + image_df['dir'] + '/' + image_df['image']
        image_df['mask_path'] = self.DataPath + image_df['dir'] + '/' + image_df['mask']
        return image_df

    def training_validation_configuration(self):
        data = self.prepare_data()
        # split the data into training and validation sets
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
        ### TRAINING SET ###
        # define the augmentation configuration
        data_augmenation = dict(rotation_range=0.2, zoom_range=0.1, horizontal_flip=True,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05, fill_mode='nearest')
        # define the data generators:
        if Augmentation:
            imagedatagen = ImageDataGenerator(rescale=1 / 255., **data_augmenation)
            maskdatagen = ImageDataGenerator(rescale=1 / 255., **data_augmenation)
        else:
            imagedatagen = ImageDataGenerator(rescale=1 / 255.)
            maskdatagen = ImageDataGenerator(rescale=1 / 255.)
        # create the training image generator
        self.train_image_generator = imagedatagen.flow_from_dataframe(dataframe=train_df,
                                                                      x_col='image_path',
                                                                      batch_size=BATCH_SIZE,
                                                                      class_mode=None,
                                                                      target_size=(ImgHieght, ImgWidth),
                                                                      seed=42,
                                                                      color_mode='rgb',
                                                                      shuffle=True)
        # create the training mask generator
        self.train_mask_generator = maskdatagen.flow_from_dataframe(dataframe=train_df,
                                                                    x_col='mask_path',
                                                                    batch_size=BATCH_SIZE,
                                                                    class_mode=None,
                                                                    target_size=(ImgHieght, ImgWidth),
                                                                    seed=42,
                                                                    color_mode='grayscale')
        #### VALIDATION SET ###
        # define the data generators:
        val_imagedatagen = ImageDataGenerator(rescale=1 / 255.)
        val_maskdatagen = ImageDataGenerator(rescale=1 / 255.)
        # create the validation image generator
        self.val_image_generator = val_imagedatagen.flow_from_dataframe(dataframe=val_df,
                                                                        x_col='image_path',
                                                                        batch_size=BATCH_SIZE,
                                                                        class_mode=None,
                                                                        target_size=(ImgHieght, ImgWidth),
                                                                        seed=42,
                                                                        color_mode='rgb')
        # create the validation mask generator
        self.val_mask_generator = val_maskdatagen.flow_from_dataframe(dataframe=val_df,
                                                                      x_col='mask_path',
                                                                      batch_size=BATCH_SIZE,
                                                                      class_mode=None,
                                                                      target_size=(ImgHieght, ImgWidth),
                                                                      seed=42,
                                                                      color_mode='grayscale')
        return self.train_image_generator.n, self.val_image_generator.n
    #

    def data_iterator(self, image_generator, mask_generator):
        while True:
            X, Y = next(image_generator), next(mask_generator)
            yield X, Y

    def data_generator(self):
        return self.data_iterator(self.train_image_generator, self.train_mask_generator), self.data_iterator(self.val_image_generator, self.val_mask_generator)

# Define UNET model in Keras in order to train it
# we define it in class type in order to use it in the training function


class UNET():
    def __init__(self, dropout_rate=0.1, batch_norm=True):
        self.ImgHieght = ImgHieght
        self.ImgWidth = ImgWidth
        self.Channels = Channels
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.model = self.build_UNET()
        self.model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
        self.model_save_path = MODEL_SAVE_PATH

    def conv2d_block(self, inputs, filters, kernel_size, batchnorm=False):
        """
        This function creates a convolutional block consisting of two convolutional layers
        and an optional batch normalization layer.
        """
        # first layer
        x = Conv2D(filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(inputs)
        x = Activation('relu')(x)
        if BachNopm_Dropout:
            x = BatchNormalization()(x)
        # second layer
        x = Conv2D(filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(x)
        x = Activation('relu')(x)
        if BachNopm_Dropout:
            x = BatchNormalization()(x)
        return x

    def build_UNET(self):
        inputs = Input((self.ImgHieght, self.ImgWidth, self.Channels))
        # first layer
        x = self.conv2d_block(inputs, 64, 3, batchnorm=self.batch_norm)
        # encoder side of the UNET
        enc1 = self.conv2d_block(x, 64, 3, batchnorm=self.batch_norm)
        pol1 = MaxPooling2D((2, 2))(enc1)
        drp1 = Dropout(self.dropout_rate)(pol1)
        # second layer
        enc2 = self.conv2d_block(drp1, 128, 3, batchnorm=self.batch_norm)
        pol2 = MaxPooling2D((2, 2))(enc2)
        drp2 = Dropout(self.dropout_rate)(pol2)
        # third layer
        enc3 = self.conv2d_block(drp2, 256, 3, batchnorm=self.batch_norm)
        pol3 = MaxPooling2D((2, 2))(enc3)
        drp3 = Dropout(self.dropout_rate)(pol3)
        # fourth layer
        enc4 = self.conv2d_block(drp3, 512, 3, batchnorm=self.batch_norm)
        pol4 = MaxPooling2D((2, 2))(enc4)
        drp4 = Dropout(self.dropout_rate)(pol4)
        # fifth layer or the bottleneck
        enc5 = self.conv2d_block(drp4, 1024, 3, batchnorm=self.batch_norm)
        # decoder side of the UNET
        # 1
        dec1 = Conv2DTranspose(512, 3, strides=2, padding='same')(enc5)
        dec2 = self.conv2d_block(concatenate([dec1, enc4]), 512, 3, batchnorm=self.batch_norm)
        dec2 = Dropout(self.dropout_rate)(dec2)
        # 2
        dec2 = Conv2DTranspose(256, 3, strides=2, padding='same')(dec2)
        dec3 = self.conv2d_block(concatenate([dec2, enc3]), 256, 3, batchnorm=self.batch_norm)
        dec3 = Dropout(self.dropout_rate)(dec3)
        # 3
        dec3 = Conv2DTranspose(128, 3, strides=2, padding='same')(dec3)
        dec4 = self.conv2d_block(concatenate([dec3, enc2]), 128, 3, batchnorm=self.batch_norm)
        dec4 = Dropout(self.dropout_rate)(dec4)
        # 4
        dec4 = Conv2DTranspose(64, 3, strides=2, padding='same')(dec4)
        dec5 = self.conv2d_block(concatenate([dec4, enc1]), 32, 3, batchnorm=self.batch_norm)
        dec5 = Dropout(self.dropout_rate)(dec5)
        # final layer
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(dec5)
        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    def train(self):
        data_train = Data_PreProcessing()
        num_train, num_valid = data_train.training_validation_configuration()
        STEP_SIZE_TRAIN = num_train / BATCH_SIZE
        STEP_SIZE_VALID = num_valid / BATCH_SIZE
        train_gen, val_gen = data_train.data_generator()
        Callbacks = [ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True, save_weights_only=True),
                     EarlyStopping(monitor='val_loss', patience=15, verbose=1),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)]
        self.history = self.model.fit_generator(train_gen,
                                                steps_per_epoch=STEP_SIZE_TRAIN,
                                                epochs=EPOCHS,
                                                validation_data=val_gen,
                                                validation_steps=STEP_SIZE_VALID,
                                                callbacks=Callbacks)

    def plot_history(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def validation(self):
        data_validation = Data_PreProcessing()
        num_train, num_valid = data_validation.training_validation_configuration()
        STEP_SIZE_TRAIN = num_train / BATCH_SIZE
        STEP_SIZE_VALID = num_valid / BATCH_SIZE
        train_gen, val_gen = data_validation.data_generator()
        self.model.load_weights(self.model_save_path)
        self.evaluated_results = self.model.evaluate_generator(val_gen, steps=STEP_SIZE_VALID)
        print('Loss: ', self.evaluated_results[0])
        print('Accuracy: ', self.evaluated_results[1])

    def plot_image_results(self, num_images=5):
        # show performance of the model on some of the images in the dataset
        data_load = Data_PreProcessing()
        imagepath_df = data_load.prepare_data()
        for i in range(0, num_images):
            idx = np.random.randint(0, len(imagepath_df))
            image_path = os.path.join(DataPath, imagepath_df['dir'].iloc[idx], imagepath_df['image'].iloc[idx])
            mask_path = os.path.join(DataPath, imagepath_df['dir'].iloc[idx], imagepath_df['mask'].iloc[idx])
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)
            img = cv2.resize(image, (self.ImgWidth, self.ImgHieght))
            msk = cv2.resize(mask, (self.ImgWidth, self.ImgHieght))

            img = img.astype(np.float32)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            #
            self.model.load_weights(self.model_save_path)
            predict = self.model.predict(img)
            # plot the original input image the original mask and the output image, and the binary prediction
            fig, ax = plt.subplots(1, 4, figsize=(20, 10))
            ax[0].imshow(image)
            ax[0].set_title('Original image')
            ax[1].imshow(mask)
            ax[1].set_title('Original mask')
            ax[2].imshow(predict[0, :, :, 0])
            ax[2].set_title('Predicted mask')
            ax[3].imshow(predict[0, :, :, 0] > 0.5)
            ax[3].set_title('Predicted mask binary')
            plt.show()


def main():
    unet = UNET()
    unet.train()
    unet.plot_history()
    unet.validation()
    unet.plot_image_results()


if __name__ == '__main__':
    main()
