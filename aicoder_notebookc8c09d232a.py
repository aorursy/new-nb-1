from google.colab import files
import zipfile
train_uploaded = files.upload()
import zipfile
with zipfile.ZipFile("train.zip", 'r') as zip_ref:
    zip_ref.extractall("./")
import matplotlib.pyplot as plt
from PIL import Image
im = Image.open('./train/100.jpg')
plt.imshow(im)
