# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from os import listdir

train_path = '../input/state-farm-distracted-driver-detection/train/'
test_path = '../input/state-farm-distracted-driver-detection/test/'

test_path_array = listdir(test_path)
matching = [s for s in test_path_array if "img_1.jpg" in s]
import pandas as pd

driver_imgs = pd.read_csv('../input/state-farm-distracted-driver-detection/driver_imgs_list.csv')
from tqdm import tqdm
def loadBatchImages(path):
    catList = listdir(path)
    loadedImages = []
    loadedLabels = []
    for cat in catList:
        if not cat.startswith('.'):
            deepPath = path+cat+"/"
            imageList = listdir(deepPath)
            for images in tqdm(imageList):
                img = deepPath + images
                loadedLabels.append(int(cat[1:]))
                loadedImages.append(img)
            
    return loadedImages, loadedLabels
loadedImages, loadedLabels = loadBatchImages(train_path)
num_classes = len(np.unique(loadedLabels))

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
labels_Hot = to_categorical(loadedLabels, num_classes = num_classes)
df= pd.DataFrame()

df['path']=loadedImages
df['labels'] = list(labels_Hot)
from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (128, 128)
core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen
from keras.applications.vgg16 import VGG16
#from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,\
                            Conv2D, MaxPool2D, MaxPooling2D
train_df = df

train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'labels', 
                            target_size = IMG_SIZE,
                            batch_size = 32)
t_x, t_y = next(train_gen)
from os.path import join, exists, expanduser
from os import listdir, makedirs

cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
pretrained_model_1 = VGG16(include_top = False, input_shape = t_x.shape[1:])
base_model = pretrained_model_1 # Topless
optimizer1 = keras.optimizers.Adam()
# Add top layer
x = base_model.output
x = Conv2D(100, kernel_size = (3,3), padding = 'valid')(x)
x = Flatten()(x)
x = Dropout(0.75)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Train top layer
for layer in base_model.layers:
    layer.trainable = False
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer1, 
              metrics=['accuracy'])
model.summary()
model.fit_generator(train_gen,steps_per_epoch = 100,epochs = 10)
import glob
from glob import glob
test_image_paths = glob('../input/state-farm-distracted-driver-detection/test/*.jpg', recursive=True)

X_test = pd.DataFrame()
X_test['path'] = test_image_paths
X_test['labels'] = X_test['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
X_test.head()
X_test['labels'] = X_test['labels'] + '.jpg'
X_test['labels'].head()
test_gen = flow_from_dataframe(core_idg, X_test, 
                             path_col = 'path',
                            y_col = 'labels', 
                            target_size = IMG_SIZE,
                            batch_size = 256) # we can use much larger batches for evaluation
print(len(X_test))
pred_Y =  model.predict_generator(test_gen, verbose = 1, steps = 312)
def do_clip(arr, mx): return np.clip(arr, (1-mx)/9, mx)

pred_Y = do_clip(pred_Y,0.93)
submission = pd.DataFrame()

submission['img'] = X_test['labels']

submission['c0'] = pred_Y[:,0]
submission['c1'] = pred_Y[:,1]
submission['c2'] = pred_Y[:,2]
submission['c3'] = pred_Y[:,3]
submission['c4'] = pred_Y[:,4]
submission['c5'] = pred_Y[:,5]
submission['c6'] = pred_Y[:,6]
submission['c7'] = pred_Y[:,7]
submission['c8'] = pred_Y[:,8]
submission['c9'] = pred_Y[:,9]

submission.to_csv("submission_05_01.csv",index = False)
print("The Submission file has been created!")
