
import numpy as np
import cv2
from math import exp
I16_BITS_MAX_VALUE = 65535
from scipy.signal import correlate2d
import imutils
from skimage import measure
import matplotlib.pyplot as plt
from imagemanager import ImageManager
from processing import AstroImageProcessing
from skimage.draw import polygon
import math

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate


def draw_contours2(image):
    label_img = label(image)
    regions = regionprops(label_img)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    regions_max = {}
    for props in regions:
        regions_max[props.area]=props
    rg_lst = list(regions_max.keys())
    rg_lst.sort(reverse=True)
    props =regions_max[rg_lst[0]]

    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)


    plt.show()


def draw_contours(image):


    contours = measure.find_contours(image, 0.2)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    thresh = image > np.percentile(image, 30) 
    max_contour=0
    cnt = {}
    for contour in contours:
        rr, cc = polygon(contour[:, 0], contour[:, 1], thresh.shape)
        area = len(rr)  # chaque pixel compte pour une unité d'aire
        if area>5:
            # Calculer le rectangle englobant et d'autres propriétés
            minr, minc, maxr, maxc = measure.regionprops(contour.astype(int))[0].bbox
            centroid = measure.regionprops(contour.astype(int))[0].centroid
            width = maxc - minc
            height = maxr - minr
            cnt[area]=(contour,centroid,width,height)

    max_cnt = list(cnt.keys())
    max_cnt.sort(reverse=True)
    print(max_cnt)
    if len(max_cnt)>3:
        max_cnt = max_cnt[0:2]
    else:
        max_cnt = [max_cnt[0]]
    print(max_cnt)
    for area in max_cnt:
        (contour,centroid,width,height) = cnt[area]
        print(f'Contour area: {area}')
        print(f'Bounding box center (y, x): {centroid}')
        print(f'Width: {width}')
        print(f'Height: {height}')

        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def show_image_3d(image):

    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')


    # Utiliser la valeur des pixels comme coordonnées Z
    z = image

    # Afficher la surface
    ax.plot_surface(x, y, z)

    # Définir les libellés des axes
    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Valeur des pixels')
    ax.view_init(56, 134, 0)
    # Afficher le graphique
    plt.show()


imager = ImageManager()
image = imager.read_image("result_elipse.png")


# Calculer les gradients le long de x et de y
gx, gy = np.gradient(image)

# Créer un maillage de coordonnées pour les positions x et y
x = np.arange(0, gx.shape[1])
y = np.arange(0, gx.shape[0])
x, y = np.meshgrid(x, y)

# Créer la figure 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Tracer la surface 3D
ax.plot_surface(x, y, np.sqrt(gx**2 + gy**2), cmap='viridis')

# Configurer les étiquettes des axes
ax.set_xlabel('Axis X')
ax.set_ylabel('Axis Y')
ax.set_zlabel('Gradient Magnitude')

# Afficher la figure
plt.show()
print(image)
print(image.max(),image.min())
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
print(minVal, maxVal, minLoc, maxLoc) 
plt.figure()
plt.imshow(image)
plt.title("Observed and expected secondary locations")
plt.show() 

show_image_3d(image)

image=(image/image.max() * 65535).astype(int)
draw_contours2(image)
  
