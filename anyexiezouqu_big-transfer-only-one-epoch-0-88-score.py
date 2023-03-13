import os

import re

import seaborn as sns

import numpy as np

import pandas as pd

import math

import tensorflow_hub as hub

from matplotlib import pyplot as plt



from sklearn import metrics

from sklearn.model_selection import train_test_split



import tensorflow as tf

import tensorflow.keras.layers as L





from kaggle_datasets import KaggleDatasets
try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)

def seed_everything(seed=0):

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'



seed = 1024

seed_everything(seed)
# For tf.dataset

AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')



# Configuration

EPOCHS = 1

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

IMAGE_SIZE = [224, 224]
def append_path(pre):

    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
sns.countplot(train['target'])
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')

valid = True

if valid:

    TRAINING_FILENAMES, VALIDATION_FILENAMES = train_test_split(

    tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec'),

    test_size=0.1, random_state=5)



CLASSES = [0,1]   
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

#     image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    image = tf.image.resize(image, IMAGE_SIZE)

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        #"class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    #label = tf.cast(example['class'], tf.int32)

    label = tf.cast(example['target'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['image_name']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALID_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images,{} vaid images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES,NUM_VALID_IMAGES, NUM_TEST_IMAGES))
def build_lrfn(lr_start=0.00001, lr_max=0.0001, 

               lr_min=0.000001, lr_rampup_epochs=20, 

               lr_sustain_epochs=0, lr_exp_decay=.8):

    lr_max = lr_max * strategy.num_replicas_in_sync



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn


# model_url = "https://tfhub.dev/google/bit/m-r50x1/1"

# # module = hub.KerasLayer("https://tfhub.dev/google/bit/m-r152x4/1")

# # module = hub.KerasLayer("https://tfhub.dev/google/bit/m-r101x1/1")

# # module = hub.KerasLayer("https://tfhub.dev/google/bit/m-r101x3/1")

# module = hub.KerasLayer(model_url)
MODELPATH = KaggleDatasets().get_gcs_path('big-transfer-models-without-top')

# module = hub.KerasLayer(f'{MODELPATH}/bit_m-r101x1_1/')

# module = hub.KerasLayer(f'{MODELPATH}/bit_m-r101x3_1/')

# module = hub.KerasLayer(f'{MODELPATH}/bit_m-r152x4_1/')

# module = hub.KerasLayer(f'{MODELPATH}/bit_m-r50x1_1/')

module = hub.KerasLayer(f'{MODELPATH}/bit_m-r50x3_1/')
lr = 0.003 * BATCH_SIZE / 512 



lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[5,10,15], 

                                                                   values=[lr, lr*0.1, lr*0.001, lr*0.0001])
with strategy.scope():

    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))

    MODELPATH = KaggleDatasets().get_gcs_path('big-transfer-models-without-top')

    module = hub.KerasLayer(f'{MODELPATH}/bit_m-r152x4_1/')

    back_bone = module

    back_bone.trainable = True

    logits = back_bone(inputs)

#     logits = tf.keras.layers.Dense(32, activation='relu', dtype='float32')(logits)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(logits)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(

        optimizer=tf.keras.optimizers.Adam(),

        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.01),

        metrics=['binary_crossentropy',tf.keras.metrics.AUC()]

    )

    model.summary()
lrfn = build_lrfn()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

class_weight = {0: 1, 1: 2}
history = model.fit(

    get_training_dataset(), 

    epochs=EPOCHS, 

    callbacks=[lr_schedule],

    steps_per_epoch=STEPS_PER_EPOCH,

    class_weight=class_weight,

    validation_data=get_validation_dataset()

)
test_ds = get_test_dataset(ordered=True)



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)
print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})

pred_df.head()
sub.head()
del sub['target']

sub = sub.merge(pred_df, on='image_name')

#sub.to_csv('submission_label_smoothing.csv', index=False)

sub.to_csv('submission.csv', index=False)

sub.head()