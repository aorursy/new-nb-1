import numpy as np

import matplotlib.pyplot as plt
import matplotlib

plt.gray()

import skimage.filters

import scipy.ndimage

import sklearn.feature_extraction
import sklearn.cluster

import skimage.feature
import skimage.transform

import PIL

import sys
for name, module in sorted(sys.modules.items()):
    if name in ['numpy', 'matplotlib', 'skimage', 'scipy', 'sklearn', 'PIL']:
        if hasattr(module, '__version__'): 
            print(name, module.__version__)
def plot_file(filename):
    image = plt.imread(filename)
    hist = np.histogram(image - image.mean(),
                        bins=np.arange(image.min(),
                                       image.max(),
                                       1/256))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    
    axes[0].imshow(image, interpolation='nearest')
    axes[0].axis('off')
    axes[0].set_title(filename[-20:])
    
    axes[1].plot(hist[1][:-1], hist[0], lw=2)
    axes[1].set_title('histogram of gray values')
    
    plt.show()
plot_file('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_red.png')
plot_file('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_green.png')
plot_file('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_blue.png')
plot_file('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_yellow.png')
def plot_cut(uuid, title, threashold_function):
    """Takes an uuid, a title for the plot and a function to apply on the image and obtain a threashold
    and plot the black and white image generated by the binary result of the inequality image > threashold"""
    
    def set_title(title, threashold):
        if np.shape(threashold) != ():
            plt.title('%s threashold=[%s]' % (title, str(np.shape(threashold))))
        else:
            plt.title('%s threashold=%0.3f' % (title, threashold))

    red    = f'../input/train/{uuid}_red.png'
    green  = f'../input/train/{uuid}_green.png'
    blue   = f'../input/train/{uuid}_blue.png'
    yellow = f'../input/train/{uuid}_yellow.png'
    
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle(title, fontsize=24)
    
    plt.subplot(1,4,1)
    image = plt.imread(red)
    threashold = threashold_function(image)
    plt.imshow(image > threashold)
    set_title('red', threashold)

    plt.subplot(1,4,2)
    image = plt.imread(green)
    threashold = threashold_function(image)
    plt.imshow(image > threashold)
    set_title('green', threashold)

    plt.subplot(1,4,3)
    image = plt.imread(blue)
    threashold = threashold_function(image)
    plt.imshow(image > threashold)
    set_title('blue', threashold)

    plt.subplot(1,4,4)
    image = plt.imread(yellow)
    threashold = threashold_function(image)
    plt.imshow(image > threashold)
    set_title('yellow', threashold)

    plt.show()
plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "zero", lambda image: 0)

plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "%5 percentile", lambda image: np.percentile(image, 5))

plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "mean", lambda image: image.mean())

plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "%95 percentile", lambda image: np.percentile(image, 95))
plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "skimage.filters.threshold_otsu", skimage.filters.threshold_otsu)
plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "skimage.filters.threshold_yen", skimage.filters.threshold_yen)
plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "skimage.filters.threshold_isodata", skimage.filters.threshold_isodata)
plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "skimage.filters.threshold_li", skimage.filters.threshold_li)
plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', 
         "skimage.filters.threshold_local(image, block_size=3)", 
         lambda image: skimage.filters.threshold_local(image, block_size=3))
plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', 
         "skimage.filters.threshold_local(image, block_size=7)", 
         lambda image: skimage.filters.threshold_local(image, block_size=7))
try:
    plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "skimage.filters.threshold_minimum", skimage.filters.threshold_minimum)
except:
    pass
plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "skimage.filters.threshold_niblack", skimage.filters.threshold_niblack)
plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "skimage.filters.threshold_sauvola", skimage.filters.threshold_sauvola)
plot_cut('002daad6-bbc9-11e8-b2bc-ac1f6b6435d0', "skimage.filters.threshold_triangle", skimage.filters.threshold_triangle)
image = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_yellow.png')

cutted = np.array(image > skimage.filters.threshold_li(image))
cutted.shape, cutted.dtype
cutted.shape[0] * cutted.shape[1]
(cutted.shape[0] * cutted.shape[1]) / 8
(cutted.shape[0] * cutted.shape[1]) / 8 / 8
packed = np.zeros((int(cutted.shape[0] / 8), int(cutted.shape[1] / 8)),
                  dtype=np.uint64)

for x in range(cutted.shape[0]):
    for y in range(cutted.shape[1]):
        new_bit = np.uint64(int(cutted[x][y]) << (x % 8) << ((y % 8) * 8))
        packed[int(x/8)][int(y/8)] = np.bitwise_or(packed[int(x/8)][int(y/8)], new_bit)

plt.figure(figsize=(15, 4))

plt.subplot(1,3,1)
plt.title(f'original {str(image.nbytes)} bytes')
plt.imshow(image)

plt.subplot(1,3,2)
plt.title(f'cutted {str(cutted.nbytes)} bytes')
plt.imshow(cutted)

plt.subplot(1,3,3)
plt.title(f'packed {str(packed.nbytes)} bytes')
plt.imshow(packed)
re = np.zeros((cutted.shape[0],cutted.shape[1]), dtype=bool)

for x in range(cutted.shape[0]):
    for y in range(cutted.shape[1]):
        re[x][y] = np.bitwise_and(np.uint64(1 << (x % 8) + (y % 8) * 8 ), packed[int(x/8)][int(y/8)])

plt.figure(figsize=(15, 4))

plt.subplot(1,3,1)
plt.title('original')
plt.imshow(image)

plt.subplot(1,3,2)
plt.title('cutted')
plt.imshow(cutted)

plt.subplot(1,3,3)
plt.title('decompressed')
plt.imshow(re)
plt.imshow(packed)
def pack_image(cutted):
    packed = np.zeros((int(cutted.shape[0] / 8), int(cutted.shape[1] / 8)),
                      dtype=np.uint64)
    for x in range(cutted.shape[0]):
        for y in range(cutted.shape[1]):
            new_bit = np.uint64(int(cutted[x][y]) << (x % 8) << ((y % 8) * 8))
            packed[int(x/8)][int(y/8)] = np.bitwise_or(packed[int(x/8)][int(y/8)], new_bit)
    return packed
re = skimage.transform.resize(cutted, (cutted.shape[0]/8,cutted.shape[1]/8), mode='reflect')
plt.imshow(re)
def pack_image2(img):
    xres, yres = np.shape(img)
    xres = int(xres / 8)
    yres = int(yres / 8)
    p = np.empty((xres,yres))
    for x in range(xres):
        for y in range(yres):
            p[x][y] = int("".join(img[x*8:(x*8)+8,y*8:(y*8)+8].flatten().astype(int).astype(str)), 2)
    return p

packed.shape, packed.dtype
packed.flatten().shape
im_noise = cutted + 0.2 * np.random.randn(*cutted.shape)
plt.title('cutted with noise')
plt.imshow(im_noise)
im_noise.dtype
cutted.dtype
hist = np.histogram(im_noise - im_noise.mean(),
                    bins=np.arange(im_noise.min(),
                                   im_noise.max(),
                                   1/256))
plt.plot(hist[1][:-1], hist[0], lw=2)
plt.title('histogram of gray values "cutted with noise" image')
hist = np.histogram(image - image.mean(),
                    bins=np.arange(image.min(),
                                   image.max(),
                                   1/256))
plt.plot(hist[1][:-1], hist[0], lw=2)
plt.title('original image histogram of gray values')
plt.figure(figsize=(15,15))
plt.subplot(1,3,1)
plt.title('cutted')
plt.imshow(cutted)

plt.subplot(1,3,2)
plt.title('binary_opening(cutted)')
open_img = scipy.ndimage.binary_opening(cutted)
plt.imshow(open_img)

plt.subplot(1,3,3)
plt.title('binary_closing(binary_opening(cutted))')
close_img = scipy.ndimage.binary_closing(open_img)
plt.imshow(close_img)
eroded_img = scipy.ndimage.binary_erosion(cutted)
reconstruct_img = scipy.ndimage.binary_propagation(eroded_img, mask=cutted)
tmp = np.logical_not(reconstruct_img)
eroded_tmp = scipy.ndimage.binary_erosion(tmp)
reconstruct_final = np.logical_not(scipy.ndimage.binary_propagation(eroded_tmp, mask=tmp))
plt.imshow(reconstruct_final)
image = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_blue.png')
cutted = image > image.mean()

graph = sklearn.feature_extraction.image.img_to_graph(image, mask=cutted)
graph.data = np.exp(-graph.data/graph.data.std())

labels = sklearn.cluster.spectral_clustering(graph, n_clusters=20, eigen_solver='arpack')
label_im = -np.ones(cutted.shape)
label_im[cutted] = labels
plt.imshow(label_im, cmap='nipy_spectral')
label_im, nb_labels = scipy.ndimage.label(image > image.mean())
plt.imshow(label_im, cmap='nipy_spectral')
np.unique(label_im).size
def disk_structure(n):
    struct = np.zeros((2 * n + 1, 2 * n + 1))
    x, y = np.indices((2 * n + 1, 2 * n + 1))
    mask = (x - n)**2 + (y - n)**2 <= n**2
    struct[mask] = 1
    return struct.astype(np.bool)

def granulometry(data, sizes=None):
    s = max(data.shape)
    if sizes is None:
        sizes = range(1, int(s/2), 2)
    granulo = [scipy.ndimage.binary_opening(data, \
        structure=disk_structure(n)).sum() for n in sizes]
    return granulo
im = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_blue.png')
mask = im > im.mean()
granulo = granulometry(mask, sizes=np.arange(1, 10, 1))
granulo
plt.figure(figsize=(12, 4.4))

plt.subplot(121)
plt.imshow(mask, cmap=plt.cm.gray)
opened = scipy.ndimage.binary_opening(mask, structure=disk_structure(10))
opened_more = scipy.ndimage.binary_opening(mask, structure=disk_structure(14))
plt.contour(opened, [0.5], colors='b', linewidths=1)
plt.contour(opened_more, [0.5], colors='r', linewidths=1)
plt.axis('off')
plt.subplot(122)
plt.plot(np.arange(1, 10, 1), granulo, 'ok', ms=8)

plt.subplots_adjust(wspace=0.02, hspace=0.15, top=0.95, bottom=0.15, left=0, right=0.95)
plt.show()
def plot_granulos(filename, sizes=np.arange(1, 10, 1)):
    im = plt.imread(filename)
    mask = im > skimage.filters.threshold_li(im)
    granulo = granulometry(mask, sizes=sizes)

    plt.figure(figsize=(12, 4.4))

    plt.subplot(121)
    plt.imshow(mask, cmap=plt.cm.gray)
    opened = scipy.ndimage.binary_opening(mask, structure=disk_structure(1))
    opened_more = scipy.ndimage.binary_opening(mask, structure=disk_structure(4))
    plt.contour(opened, [0.5], colors='b', linewidths=1)
    plt.contour(opened_more, [0.5], colors='r', linewidths=1)
    plt.axis('off')
    plt.subplot(122)
    plt.plot(sizes, granulo, 'ok', ms=8)

    plt.subplots_adjust(wspace=0.02, hspace=0.15, top=0.95, bottom=0.15, left=0, right=0.95)
    plt.show()
plot_granulos('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_red.png')
plot_granulos('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_green.png')
plot_granulos('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_blue.png')
plot_granulos('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_yellow.png')
image = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_blue.png')

s = np.linspace(0, 2*np.pi, 400)
x = 150 + 50*np.cos(s)
y = 150 + 50*np.sin(s)
init = np.array([x, y]).T

filtered = skimage.filters.gaussian(image, 2)
plt.subplot(1,2,1)
plt.imshow(image)

plt.subplot(1,2,2)
plt.imshow(filtered)
snake = skimage.segmentation.active_contour(filtered,
                                            init,
                                            alpha=0.002,
                                            beta=10,
                                            gamma=0.01)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(image)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, image.shape[1], image.shape[0], 0])

plt.show()
image = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_green.png')

blobs_log = skimage.feature.blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * (2 ** .5)

blobs_dog = skimage.feature.blob_dog(image, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * (2 ** .5)

blobs_doh = skimage.feature.blob_doh(image, max_sigma=30, threshold=.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image, interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)

plt.tight_layout()
plt.show()
image = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_red.png')
edges = skimage.feature.canny(image)

plt.imshow(edges)
plt.title('Canny detector');
fill_image = scipy.ndimage.binary_fill_holes(edges)

plt.imshow(fill_image)
plt.title('filling the holes');
red    = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_red.png')
green  = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_green.png')
blue   = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_blue.png')
yellow = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_yellow.png')
def to_rgba(img):
    (nro_channels, resolutionx, resolutiony) = np.shape(img)
    r = [[[1, 0, 0, img[0][i][j]] for i in range(resolutionx)] for j in range(resolutiony)]
    g = [[[0, 1, 0, img[1][i][j]] for i in range(resolutionx)] for j in range(resolutiony)]
    b = [[[0, 0, 1, img[2][i][j]] for i in range(resolutionx)] for j in range(resolutiony)]
    y = [[[1, 1, 0, img[3][i][j]] for i in range(resolutionx)] for j in range(resolutiony)]
    return np.array([r,g,b,y])
plt.figure(figsize=(15,15))
for i in range(4):
    plt.imshow(r[i])
def to_rgba2(img):
    r = np.transpose(np.vectorize(lambda x: (1,0,0,x))(img[0]))
    g = np.transpose(np.vectorize(lambda x: (0,1,0,x))(img[1]))
    b = np.transpose(np.vectorize(lambda x: (0,0,1,x))(img[2]))
    y = np.transpose(np.vectorize(lambda x: (1,1,0,x))(img[3]))
    return np.array([r,g,b,y])
plt.figure(figsize=(15,15))
for i in range(4):
    plt.imshow(r[i])
red     = PIL.Image.open('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_red.png')
green   = PIL.Image.open('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_green.png')
blue    = PIL.Image.open('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_blue.png')
yellow  = PIL.Image.open('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_yellow.png')
type(red)
red.format
red.mode
S1 = PIL.ImageChops.blend(red,green,0.5)
S2 = PIL.ImageChops.blend(blue,yellow,0.5)
S3 = PIL.ImageChops.blend(S1,S2,0.5)
S3
plt.imshow(np.asarray(S3))
red.convert('RGB')
red.mode
rgb = PIL.Image.merge('RGB', (red, green, blue))
rgb
y = PIL.Image.merge('RGB', (yellow, yellow, PIL.Image.new('L', (yellow.width, yellow.height))))
y
rgby = PIL.ImageChops.add(rgb, y)
rgby
image = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_blue.png')*255
image = image.astype(int)
elevation_map = skimage.filters.sobel(image/255)

plt.imshow(elevation_map)
plt.title('elevation map')
markers = np.zeros_like(image)
markers[image > 0] = 1
markers[image > 25] = 2
markers[image > 30] = 3
markers[image > 35] = 4

plt.imshow(markers, cmap='nipy_spectral')
plt.title('markers')
segmentation = skimage.morphology.watershed(elevation_map, markers)

plt.imshow(segmentation)
plt.title('segmentation')
segmentation = scipy.ndimage.binary_fill_holes(segmentation - 1)
labeled, _ = scipy.ndimage.label(segmentation)
image_label_overlay = skimage.color.label2rgb(labeled, image=image/255)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
axes[1].imshow(image_label_overlay, interpolation='nearest')

plt.tight_layout()

plt.show()
np.unique(labeled).size
image = plt.imread('../input/train/002daad6-bbc9-11e8-b2bc-ac1f6b6435d0_blue.png')

fig = plt.figure(figsize=(10, 6))
ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
ax1.imshow(image)

ax2 = plt.subplot2grid((2, 4), (1, 0))
ax3 = plt.subplot2grid((2, 4), (1, 1))
ax4 = plt.subplot2grid((2, 4), (1, 2))
ax5 = plt.subplot2grid((2, 4), (1, 3))

# apply threshold
thresh = skimage.filters.threshold_otsu(image)
bw = skimage.morphology.closing(image > thresh, skimage.morphology.square(2))
ax2.imshow(bw)

# remove artifacts connected to image border
cleared = skimage.segmentation.clear_border(bw)
ax3.imshow(cleared)

# label image regions
label_image = skimage.measure.label(cleared)
ax4.imshow(label_image)
image_label_overlay = skimage.color.label2rgb(label_image, image=image)


#fig, ax = plt.subplots(figsize=(10, 6))
ax5.imshow(image_label_overlay)

for region in skimage.measure.regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        ax1.text(int(region.centroid[1]), int(region.centroid[0]), region.label)
        minr, minc, maxr, maxc = region.bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(rect)

ax1.set_axis_off()
plt.tight_layout()
plt.show()