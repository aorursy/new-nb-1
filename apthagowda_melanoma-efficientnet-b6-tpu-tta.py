import re

import cv2

import math

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split

from kaggle_datasets import KaggleDatasets



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras import optimizers

import efficientnet.tfkeras as efn
use_knockknock = True #set this to false if you are not using knockknock.

if use_knockknock:

    !pip install knockknock

    from knockknock import telegram_sender 

    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()

    token = user_secrets.get_secret("token") #kaggle secret token

    chat_id = user_secrets.get_secret("chat_id") #kaggle secret chat_id
AUTO = tf.data.experimental.AUTOTUNE



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()

    

print("REPLICAS: ", strategy.num_replicas_in_sync)



DATASET = '512x512-melanoma-tfrecords-70k-images'

GCS_PATH = KaggleDatasets().get_gcs_path(DATASET)
SEED = 42

BATCH_SIZE = 32 * strategy.num_replicas_in_sync

SIZE = [512,512]

LR = 0.00004

EPOCHS = 12

WARMUP = 5

WEIGHT_DECAY = 0

LABEL_SMOOTHING = 0.05

TTA = 4
def seed_everything(SEED):

    np.random.seed(SEED)

    tf.random.set_seed(SEED)



seed_everything(SEED)

train_filenames = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')

test_filenames = tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')
train_filenames,valid_filenames = train_test_split(train_filenames,test_size = 0.2,random_state = SEED)
def decode_image(image):

    image = tf.image.decode_jpeg(image, channels=3) 

    image = tf.cast(image, tf.float32)/255.0

    image = tf.reshape(image, [*SIZE, 3])

    return image



def data_augment(image, label=None, seed=SEED):

    image = tf.image.rot90(image,k=np.random.randint(4))

    image = tf.image.random_flip_left_right(image, seed=seed)

    image = tf.image.random_flip_up_down(image, seed=seed)

    if label is None:

        return image

    else:

        return image, label



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "target": tf.io.FixedLenFeature([], tf.int64),  }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['target'], tf.int32)

    return image, label 



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string), }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    image_name = example['image_name']

    return image, image_name



def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False



    dataset = (tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

              .with_options(ignore_order)

              .map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO))

            

    return dataset



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



def plot_transform(num_images):

    plt.figure(figsize=(30,10))

    x = load_dataset(train_filenames, labeled=False)

    image,_ = iter(x).next()

    for i in range(1,num_images+1):

        plt.subplot(1,num_images+1,i)

        plt.axis('off')

        image = data_augment(image=image)

        plt.imshow(image)
plot_transform(7)
train_dataset = (load_dataset(train_filenames, labeled=True)

    .map(data_augment, num_parallel_calls=AUTO)

    .shuffle(SEED)

    .batch(BATCH_SIZE,drop_remainder=True)

    .repeat()

    .prefetch(AUTO))



valid_dataset = (load_dataset(valid_filenames, labeled=True)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO))
with strategy.scope():

    model = tf.keras.Sequential([

        efn.EfficientNetB6(input_shape=(*SIZE, 3),weights='imagenet',pooling='avg',include_top=False),

        Dense(1, activation='sigmoid')

    ])

        

    model.compile(

        optimizer='adam',

        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = LABEL_SMOOTHING),

        metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=0.5):

    """

    Modified version of the get_cosine_schedule_with_warmup from huggingface.

    (https://huggingface.co/transformers/_modules/transformers/optimization.html#get_cosine_schedule_with_warmup)



    Create a schedule with a learning rate that decreases following the

    values of the cosine function between 0 and `pi * cycles` after a warmup

    period during which it increases linearly between 0 and 1.

    """



    def lrfn(epoch):

        if epoch < num_warmup_steps:

            return (float(epoch) / float(max(1, num_warmup_steps))) * lr

        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr



    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



lr_schedule= get_cosine_schedule_with_warmup(lr=LR,num_warmup_steps=WARMUP,num_training_steps=EPOCHS)
if use_knockknock == True:

    @telegram_sender(token=token, chat_id=int(chat_id))

    def train():

        STEPS_PER_EPOCH = count_data_items(train_filenames) // BATCH_SIZE

        history = model.fit(

            train_dataset, 

            epochs=EPOCHS, 

            callbacks=[lr_schedule],

            steps_per_epoch=STEPS_PER_EPOCH,

            validation_data=valid_dataset)



        string = 'Train acc:{:.4f} Train loss:{:.4f} AUC: {:.4f}, Val acc:{:.4f} Val loss:{:.4f} Val AUC: {:.4f}'.format( \

            model.history.history['accuracy'][-1],model.history.history['loss'][-1],\

            model.history.history['auc'][-1],\

            model.history.history['val_accuracy'][-1],model.history.history['val_loss'][-1],\

            model.history.history['val_auc'][-1])



        return string

else:

    def train():

        STEPS_PER_EPOCH = count_data_items(train_filenames) // BATCH_SIZE

        history = model.fit(

            train_dataset, 

            epochs=EPOCHS, 

            callbacks=[lr_schedule],

            steps_per_epoch=STEPS_PER_EPOCH,

            validation_data=valid_dataset)

        

        return
train()
def display_training_curves(training, validation, title, subplot):

    """

    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

    """

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(20,15), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])
display_training_curves(

    model.history.history['loss'], 

    model.history.history['val_loss'], 

    'loss', 311)

display_training_curves(

    model.history.history['accuracy'], 

    model.history.history['val_accuracy'], 

    'accuracy', 312)

display_training_curves(

    model.history.history['auc'], 

    model.history.history['val_auc'], 

    'accuracy', 313)
num_test_images = count_data_items(test_filenames)

submission_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

for i in range(TTA):

    test_dataset = (load_dataset(test_filenames, labeled=False,ordered=True)

    .map(data_augment, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE))

    test_dataset_images = test_dataset.map(lambda image, image_name: image)

    test_dataset_image_name = test_dataset.map(lambda image, image_name: image_name).unbatch()

    test_ids = next(iter(test_dataset_image_name.batch(num_test_images))).numpy().astype('U')

    test_pred = model.predict(test_dataset_images, verbose=1)

    pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(test_pred)})

    temp = submission_df.copy()

    del temp['target']

    submission_df['target'] += temp.merge(pred_df,on="image_name")['target']/TTA
submission_df.to_csv('submission.csv', index=False)

pd.Series(np.round(submission_df['target'].values)).value_counts()