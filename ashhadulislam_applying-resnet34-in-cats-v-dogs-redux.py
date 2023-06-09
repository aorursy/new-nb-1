
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import glob
import shutil
import os
import gc
import pathlib
print(os.listdir("../input/"))
# Any results you write to the current directory are saved as output.
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
torch.cuda.is_available()
torch.backends.cudnn.enabled
dog_indexes = []
cat_indexes = []
count = 0
cat_count = 1
dog_count = 1
import zipfile
path_to_zip_file="../input/dogs-vs-cats-redux-kernels-edition/train.zip"
directory_to_extract_to="../output/data"
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)
import zipfile
path_to_zip_file="../input/dogs-vs-cats-redux-kernels-edition/test.zip"
directory_to_extract_to="../output/data"
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)
for name in os.listdir('../output/data/train/'):
    if 'cat' in name:
        cat_indexes.append(name.split('.')[1])
    if 'dog' in name:
        dog_indexes.append(name.split('.')[1])
print ('Dog!\n',len(dog_indexes), '\nCat!\n', len(cat_indexes))
cat_val_list = random.sample(cat_indexes, 2000)
dog_val_list = random.sample(dog_indexes, 2000)
os.makedirs('../working/dogcats/valid/cats/')
os.makedirs('../working/dogcats/valid/dogs/')
os.makedirs('../working/dogcats/train/cats/')
os.makedirs('../working/dogcats/train/dogs/')
os.makedirs('../working/dogcats/test/')
train_dir = "../output/data/train/"
test_dir = "../output/data/test/"
cat_train_dir = "../working/dogcats/train/cats/"
cat_valid_dir = "../working/dogcats/valid/cats/"
dog_train_dir = "../working/dogcats/train/dogs/"
dog_valid_dir = "../working/dogcats/valid/dogs/"
dogcats_test = "../working/dogcats/test/"
PATH = "../working/dogcats/"
sz=224
for jpgfile in iglob(os.path.join(train_dir, "cat*.jpg")):
    if jpgfile.split('.')[3] in cat_val_list:
        shutil.copy(jpgfile, cat_valid_dir)
    else:
        shutil.copy(jpgfile, cat_train_dir)

for jpgfile in iglob(os.path.join(train_dir, "dog*.jpg")):
    if jpgfile.split('.')[3] in dog_val_list:
        shutil.copy(jpgfile, dog_valid_dir)
    else:
        shutil.copy(jpgfile, dog_train_dir)
        
for jpgfile in iglob(os.path.join(test_dir, "*.jpg")):
    shutil.copy(jpgfile, dogcats_test)
cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

os.makedirs('/tmp/.torch/models/')
gc.collect()
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
arch=resnet34

print(PATH)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, test_name='test')
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(1e-2, 1)
learn.precompute=False
learn.fit(1e-2, 3, cycle_len=1)
learn.sched.plot_lr()
gc.collect()
learn.unfreeze()
lr=np.array([1e-4,1e-3,1e-2])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_lr()
gc.collect()
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y)
preds = np.argmax(probs, axis=1)
probs = probs[:,1]
temp = learn.predict(is_test=True)
temp.shape
pred_test = np.argmax(temp, axis=1)
pred_test[:20]
probs = np.exp(temp[:,1])
probs[:10]
print(PATH)
os.listdir(f'{PATH}test')[:4]
submission = pd.DataFrame({'id':os.listdir(f'{PATH}test'), 'label':probs})
submission['id'] = submission['id'].map(lambda x: x.split('.')[0])
submission['id'] = submission['id'].astype(int)
submission = submission.sort_values('id')
submission.to_csv('../working/output.csv', index=False)
