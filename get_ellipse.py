import numpy as np
import cv2
from matplotlib import pyplot as plt

import streamlit as st
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
import math

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

def apply_mean_mask_subtraction(image, k_size=3):

    processed_image = image.copy()
    # Créer un noyau moyen
    kernel = np.ones((k_size, k_size), np.float32) / (k_size * k_size)

    # Appliquer le filtre moyenneur
    mean_filtered = cv2.filter2D(image, -1, kernel)

    # Soustraire le résultat filtré de l'image originale
    processed_image = cv2.subtract(processed_image, mean_filtered)

    return processed_image



def draw_contours(image):
    label_img = label(image)
    regions = regionprops(label_img)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    regions_max = {}
    for props in regions:
        regions_max[props.area]=props
    rg_lst = list(regions_max.keys())
    rg_lst.sort(reverse=True)
    for ind in rg_lst:
        props =regions_max[ind]

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


    st.pyplot(fig)

@st.cache_data
def load_image():
    image = np.load("spatial.npy")
    spatial_elipse = image/image.max()
    spatial_elipse = np.absolute(apply_mean_mask_subtraction(spatial_elipse,3))
    return spatial_elipse



spatial_elipse_initial = load_image()
h,w = spatial_elipse_initial.shape
spatial_elipse=spatial_elipse_initial[1:h-1,1:w-1]*65535


st.title('# Covid data exploration')

st.sidebar.header("Input Parameters")
max_value = st.sidebar.slider('Max filtering', 0, 0, int(spatial_elipse.mean()*10))
min_value = st.sidebar.slider('Min filtering', 0, 0, int(spatial_elipse.mean()*9))
st.sidebar.caption(f"Filter Max : {max_value}")
st.sidebar.caption(f"Filter Min : {min_value}")
st.sidebar.caption(f"Image min : {spatial_elipse.min()}")
st.sidebar.caption(f"Image max : {spatial_elipse.max()}")
st.sidebar.caption(f"Image mean : {spatial_elipse.mean()}")

fig, ax = plt.subplots()
spatial_elipse_filtered = np.clip(spatial_elipse, min_value,max_value)
spatial_elipse_filtered = spatial_elipse_filtered/spatial_elipse_filtered.max() * 65535
ax.imshow(spatial_elipse_filtered)
st.pyplot(fig)



image=(spatial_elipse_filtered/spatial_elipse_filtered.max() * 65535).astype(int)
cnt = draw_contours(spatial_elipse_filtered)


