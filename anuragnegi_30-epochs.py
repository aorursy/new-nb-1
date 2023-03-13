import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns




np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





sns.set(style='white', context='notebook', palette='deep')
train=pd.read_csv('../input/Kannada-MNIST/train.csv')

test=pd.read_csv('../input/Kannada-MNIST/test.csv')

sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
print('The Train  dataset has {} rows and {} columns'.format(train.shape[0],train.shape[1]))

print('The Test  dataset has {} rows and {} columns'.format(test.shape[0],test.shape[1]))
X_train=train.drop('label',axis=1)

Y_train=train.label
test=test.drop('id',axis=1)
X_train=X_train/255

test=test/255
X_train=X_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)
print('The shape of train set now is',X_train.shape)

print('The shape of test set now is',test.shape)
Y_train=to_categorical(Y_train)
X_train,X_test,y_train,y_test=train_test_split(X_train,Y_train,random_state=42,test_size=0.15)
plt.imshow(X_train[0][:,:,0])
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer=RMSprop(lr=0.001,rho=0.9,decay=0.0)
model.compile(optimizer=optimizer,loss=['categorical_crossentropy'],metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs=30

batch_size=64
# Fit the model

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
fig,ax=plt.subplots(2,1)

fig.set

x=range(1,1+epochs)

ax[0].plot(x,history.history['loss'],color='red')

ax[0].plot(x,history.history['val_loss'],color='blue')



ax[1].plot(x,history.history['acc'],color='red')

ax[1].plot(x,history.history['val_acc'],color='blue')

ax[0].legend(['trainng loss','validation loss'])

ax[1].legend(['trainng acc','validation acc'])

plt.xlabel('Number of epochs')

plt.ylabel('accuracy')
y_pre_test=model.predict(X_test)

y_pre_test=np.argmax(y_pre_test,axis=1)

y_test=np.argmax(y_test,axis=1)
test=pd.read_csv('../input/Kannada-MNIST/test.csv')

test_id=test.id



test=test.drop('id',axis=1)

test=test/255

test=test.values.reshape(-1,28,28,1)
y_pre=model.predict(test)     ##making prediction

y_pre=np.argmax(y_pre,axis=1) 
sample_sub['label']=y_pre

sample_sub.to_csv('submission.csv',index=False)
sample_sub.head()