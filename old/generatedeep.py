import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import splitext, isfile, isdir
from imagemanager import ImageManager
from processing import AstroImageProcessing
from pyplot_utils import show_image, show_image_3d

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

def load_speckle_images(image_files):
    imager = ImageManager()
    tab =  [roi for roi in [AstroImageProcessing.find_roi(imager.read_image(file)) for file in image_files] if roi is not None]
    max_w = 0
    max_h = 0
    for im in tab:
        (h, w) = im.shape
        max_w = max(w, max_w)
        max_h = max(h, max_h)

    output = []
    for index, im in enumerate(tab):
        if tab[index] is not None and len(tab[index]>0):
            output.append(AstroImageProcessing.resize_with_padding(im, max_w, max_h))
    return output


dir='images-etalons/'

image_files=[]
for file in listdir(dir):
    file_path = dir + '/' + file
    if isfile(file_path) and splitext(file_path)[1]=='.fit':
        image_files.append(file_path)
images = load_speckle_images(image_files)
result = []

for i in range(5):
    im = cv2.resize(images[i],(256,256))
    im = np.expand_dims(im, axis=-1)

    result.append(im)

result = np.array(result)
np.save('test.npy',result)
print(result.shape)
plt.imshow(result[0])
plt.show()