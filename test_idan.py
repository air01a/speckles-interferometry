from idan import idan

import cv2
import numpy as np
from matplotlib import pyplot as plt
from imagemanager import ImageManager


imager = ImageManager()


path =  "imagesrepo/a1799/a1799_f1953.fit"
im = imager.read_image(path)

im2 = idan(im)
plt.imshow(im2.real)
plt.show()