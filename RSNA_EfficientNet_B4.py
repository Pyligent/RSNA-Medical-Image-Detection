#!/usr/bin/env python
# coding: utf-8

# ## EfficientNet B4 model
# 
# 
# **Due to GPU quota is only 30 hours/per week on Kaggle, each training need 15+ hours, so the notebook cann't commiting(otherwise will exceeding the quota), only download the csv files to submit**
# 
# 
# 

# In[ ]:


# Install EfficentNet 
get_ipython().system('pip install efficientnet')
get_ipython().system('pip install iterative-stratification')


# In[ ]:



import efficientnet.keras as efn 
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# In[ ]:


import numpy as np
import pandas as pd
import pydicom
import os
import collections
import sys
import glob
import random
import cv2
import tensorflow as tf
import multiprocessing

from math import ceil, floor
from copy import deepcopy
from tqdm import tqdm
from imgaug import augmenters as iaa

import keras
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, load_model
from keras.utils import Sequence
from keras.losses import binary_crossentropy
from keras.optimizers import Adam


# 

# ### Model Parameters Setup

# In[ ]:


# Setting the parameters:
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


input_image_width = 256
input_image_height = 256

input_image_shape = (input_image_height,input_image_width,3)

test_size = 0.01
batch_size = 16
train_batch_size = 16
valid_batch_size  = 32

# Setting the Path 
path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'
train_img_path  = path + 'stage_2_train/'
test_img_path = path + 'stage_2_test/'

# Dataset Filenames
train_dataset_fns = path + 'stage_2_train.csv'
test_dataset_fns = path + 'stage_2_sample_submission.csv'




# In[ ]:


dup_image_list = [56346, 56347, 56348, 56349,
                            56350, 56351, 1171830, 1171831,
                            1171832, 1171833, 1171834, 1171835,
                            3705312, 3705313, 3705314, 3705315,
                            3705316, 3705317, 3842478, 3842479,
                            3842480, 3842481, 3842482, 3842483 ]


# ### load the dataset

# In[ ]:


def train_dataset_loader(filename):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    df = df.drop(index = dup_image_list)
    df = df.reset_index(drop = True)    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    return df


def test_dataset_loader(filename):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    return df



# In[ ]:


train_df = train_dataset_loader(train_dataset_fns)
test_df = test_dataset_loader(test_dataset_fns)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# ### Data EDA and Cleaning

# In[ ]:


def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def window_image(dcm, window_center, window_width):    
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    
    # Resize
    img = cv2.resize(img, SHAPE[:2], interpolation = cv2.INTER_LINEAR)
   
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img

def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)
    
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)
    return bsb_img

def _read(path, SHAPE):
    dcm = pydicom.dcmread(path)
    try:
        img = bsb_window(dcm)
    except:
        img = np.zeros(SHAPE)
    return img


# In[ ]:


def window_with_correction(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img

def window_without_correction(dcm, window_center, window_width):
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img

def window_testing(img, window):
    brain_img = window(img, 40, 80)
    subdural_img = window(img, 80, 200)
    soft_img = window(img, 40, 380)
    
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    return bsb_img


# example of a "bad data point" (i.e. (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100) == True)
import matplotlib.pyplot as plt
dicom = pydicom.dcmread(train_img_path + train_df.index[101] + ".dcm")

fig, ax = plt.subplots(1, 2)

ax[0].imshow(window_testing(dicom, window_without_correction), cmap=plt.cm.bone);
ax[0].set_title("original")
ax[1].imshow(window_testing(dicom, window_with_correction), cmap=plt.cm.bone);
ax[1].set_title("corrected");


# ### Random image augmentation

# In[ ]:


# Image Augmentation
sometimes = lambda aug: iaa.Sometimes(0.25, aug)
     
augmentation = iaa.Sequential([ iaa.Fliplr(0.25),
                                iaa.Flipud(0.10),
                                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                                iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0, 0.5))),# Strengthen or weaken the contrast in each image.
                                iaa.ContrastNormalization((0.75, 1.5)),
                                sometimes(iaa.Crop(px=(0, 25), keep_size = True, sample_independently = False))   
                            ], random_order = True)       
        


# Generators
class DataGenerator_Train(keras.utils.Sequence):
    def __init__(self, dataset, labels, batch_size = batch_size, image_shape = input_image_shape, image_path = train_img_path, augment = False, *args, **kwargs):
        self.dataset = dataset
        self.ids = dataset.index
        self.labels = labels
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.image_path = train_img_path
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X, Y = self.__data_generation(indices)
        return X, Y

    def augmentor(self, image):
        augment_img = augmentation        
        image_aug = augment_img.augment_image(image)
        return image_aug

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))
        np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        X = np.empty((self.batch_size, *self.image_shape))
        Y = np.empty((self.batch_size, 6), dtype=np.float32)
        
        for i, index in enumerate(indices):
            ID = self.ids[index]
            image = _read(self.image_path+ID+".dcm", self.image_shape)
            if self.augment:
                X[i,] = self.augmentor(image)
            else:
                X[i,] = image
            Y[i,] = self.labels.iloc[index].values        
        return X, Y
    
class DataGenerator_Test(keras.utils.Sequence):
    def __init__(self, dataset, labels, batch_size = batch_size, image_shape = input_image_shape, image_path = test_img_path, *args, **kwargs):
        self.dataset = dataset
        self.ids = dataset.index
        self.labels = labels
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.image_path = image_path
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(indices)
        return X

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))
    
    def __data_generation(self, indices):
        X = np.empty((self.batch_size, *self.image_shape))
        
        for i, index in enumerate(indices):
            ID = self.ids[index]
            image = _read(self.image_path+ID+".dcm", self.image_shape)
            X[i,] = image              
        return X


# Import the training and test datasets.

# - oversample the minority class 'epidural' 

# In[ ]:


# Oversampling
epidural_df = train_df[train_df.Label['epidural'] == 1]
train_oversample_df = pd.concat([train_df, epidural_df])
train_df = train_oversample_df

# Summary
print('Train Shape: {}'.format(train_df.shape))
print('Test Shape: {}'.format(test_df.shape))


# ### EfficientNet model

# In[ ]:


def predictions(test_df, model):    
    test_preds = model.predict_generator(DataGenerator_Test(test_df, None, 5, input_image_shape, test_img_path), verbose = 1)
    return test_preds[:test_df.iloc[range(test_df.shape[0])].shape[0]]

def ModelCheckpointFull(model_name):
    return ModelCheckpoint(model_name, 
                            monitor = 'val_loss', 
                            verbose = 1, 
                            save_best_only = False, 
                            save_weights_only = True, 
                            mode = 'min', 
                            period = 1)

# Create Model
def create_model():
    K.clear_session()
    
    base_model =  efn.EfficientNetB4(weights = 'imagenet', include_top = False, pooling = 'avg', input_shape = input_image_shape)
    x = base_model.output
    x = Dropout(0.2)(x)
    y_pred = Dense(6, activation = 'sigmoid')(x)

    return Model(inputs = base_model.input, outputs = y_pred)


# ### Multi-Labels Train/Valid Dataset Split

# In[ ]:


# Submission Placeholder
submission_predictions = []


Multi_Stratified_split = MultilabelStratifiedShuffleSplit(n_splits = 10, test_size = test_size, random_state = seed)
X = train_df.index
Y = train_df.Label.values

# Get train and test index
Multi_Stratified_splits = next(Multi_Stratified_split.split(X, Y))
train_idx = Multi_Stratified_splits[0]
valid_idx = Multi_Stratified_splits[1]


# 

# In[ ]:


# Loop through Folds of Multi Label Stratified Split

for epoch in range(0, 4):
    print('=========== EPOCH {}'.format(epoch))

    # Shuffle Train data
    np.random.shuffle(train_idx)
    print(train_idx[:5])    
    print(valid_idx[:5])

    # Create Data Generators for Train and Valid
    data_generator_train = DataGenerator_Train(train_df.iloc[train_idx], 
                                                train_df.iloc[train_idx], 
                                                train_batch_size, 
                                                input_image_shape,
                                                augment = True)
    data_generator_val = DataGenerator_Train(train_df.iloc[valid_idx], 
                                             train_df.iloc[valid_idx], 
                                             valid_batch_size, 
                                             input_image_shape,
                                             augment = False)

    # Create Model
    model = create_model()
    
    # Full Training Model
    for base_layer in model.layers[:-1]:
        base_layer.trainable = True
    steps = int(len(data_generator_train) / 6)
    LR = 0.0001

    if epoch != 0:
        # Load Model Weights
        model.load_weights('model.h5')    

    model.compile(optimizer = Adam(learning_rate = LR), 
                  loss = 'binary_crossentropy',
                  metrics = ['acc', tf.keras.metrics.AUC()])
    
    # Train Model
    model.fit_generator(generator = data_generator_train,
                        validation_data = data_generator_val,
                        steps_per_epoch = steps,
                        epochs = 1,
                        callbacks = [ModelCheckpointFull('model.h5')],
                        verbose = 1)
    
    # Starting with the 6th epoch we create predictions for the test set on each epoch
    if epoch >= 1:
        preds = predictions(test_df, model)
        submission_predictions.append(preds)


# ### Ensemble and average all submission_predictions.

# In[ ]:


test_df.iloc[:, :] = np.average(submission_predictions, axis = 0, weights = [2**i for i in range(len(submission_predictions))])
test_df = test_df.stack().reset_index()
test_df.insert(loc = 0, column = 'ID', value = test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])
test_df = test_df.drop(["Image", "Diagnosis"], axis=1)
test_df.to_csv('submission.csv', index = False)
print(test_df.head(12))


# In[ ]:


from IPython.display import FileLink, FileLinks
FileLink('submission.csv')

