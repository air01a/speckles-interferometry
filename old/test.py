import numpy as np
import cv2
from matplotlib import pyplot as plt

from os import listdir
from os.path import splitext, isfile, isdir
from imagemanager import ImageManager
from processing import AstroImageProcessing
from pyplot_utils import show_image, show_image_3d, show_images

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


path = 'imagesrepo/stf2380'
image_files=[]
for file in listdir(path):
    file_path = path + '/' + file
    if isfile(file_path) and splitext(file_path)[1]=='.fit':
        image_files.append(file_path)
speckle_images = load_speckle_images(image_files)

processed=[]
for im in speckle_images:
    image = im
    #image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image = ((image-image.min())/(image.max()-image.min())*255).astype(np.uint8)
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [sorted_contours[0]]
    #image = cv2.drawContours(image, contours, -1, (255), 3)
    ellipse = cv2.fitEllipse(contours[0])
    cv2.ellipse(image,ellipse,(0,255,0),2)
    processed.append(image)

show_images(processed,max_images=30)
#for im in speckle_images:
#    show_image_3d(im)
im = np.absolute(AstroImageProcessing.apply_mean_mask_subtraction(np.mean(speckle_images, axis=0),5))
show_image_3d(im)
show_image(im)
