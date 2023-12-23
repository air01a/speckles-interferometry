import numpy as np
import cv2
from matplotlib import pyplot as plt

import streamlit as st
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.filters import sobel
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import math
from skimage.draw import line_aa
from processing import AstroImageProcessing
from pyplot_utils import show_image, show_image_3d
import json
from sklearn.preprocessing import StandardScaler
from idan import idan
from sklearn.cluster import KMeans
from skimage.filters import rank
from skimage.morphology import disk
from skimage.filters import rank
from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration


def detect_and_remove_lines(image):

    (h,w) = image.shape
    for i in range(0,h):
        image[int(h/2), i] = (image[int(h/2)-1, i] + image[int(h/2)+1, i])/2
        image[i, int(h/2)] = (image[i,int(h/2)-1] + image[i,int(h/2)+1])/2
    return image

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
    rg_lst=[rg_lst[0]]
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
def load_image(file):
    image = np.load(file)
    spatial_elipse = image/image.max()
    print (spatial_elipse.shape)
    return spatial_elipse

def calculer_cercle_circonscrit(pts):
    print(pts[0,0])
    x1, y1 = pts[0,0]
    x2, y2 = pts[1,0]
    x3, y3 = pts[2,0]

    # Calcul des médiatrices pour les segments [P1P2] et [P1P3]
    ma = (y2 - y1) / (x2 - x1)
    mb = (y3 - y1) / (x3 - x1)
    print("ma/mb",ma,mb)

    # Calcul du centre du cercle (xc, yc)
    xc = (ma * mb * (y1 - y3) + mb * (x1 + x2) - ma * (x1 + x3)) / (2 * (mb - ma))
    yc = -1/ma * (xc - (x1 + x2) / 2) + (y1 + y2) / 2
    print("xc/yc",xc,yc)

    # Calcul du rayon
    rayon = np.sqrt((xc - x1)**2 + (yc - y1)**2)

    return (int(xc), int(yc)), int(rayon)

def angle_avec_reference(centre, reference, point):
    vecteur1 = reference - centre
    vecteur2 = point - centre
    unit_vector_1 = vecteur1 / np.linalg.norm(vecteur1)
    unit_vector_2 = vecteur2 / np.linalg.norm(vecteur2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

print("--------------")
st.title('# Speckle interferometry analysis')

st.sidebar.header("Input Parameters")
max_filter_value = st.sidebar.slider('Max filtering', 300,65535, 300, )
uploaded_file = st.file_uploader("Choose a file",type=['npy'])
max_value = st.sidebar.slider('Max filtering', 0,max_filter_value, 300, )
min_value = st.sidebar.slider('Min filtering', 0, 8000, 0)
mean_filter = st.sidebar.slider('Mean filtering', 3, 14, 3, 2)
level = st.sidebar.slider('Contour level', 0.0, 1.0, 0.8, 0.1)

if uploaded_file is not None:
    st.header(uploaded_file.name)
    st.divider()
    name = uploaded_file.name.split('.')[0]
    spatial_elipse_initial = load_image(uploaded_file)

    h,w = spatial_elipse_initial.shape
    
    st.sidebar.caption(f"Filter Max : {max_value}")
    st.sidebar.caption(f"Filter Min : {min_value} / {min_value/65535}")
    st.sidebar.caption(f"Mean filter : {mean_filter}")
    st.sidebar.caption(f"Contour level : {level}")
    st.sidebar.caption(f"Image min : {spatial_elipse_initial.min()}")
    st.sidebar.caption(f"Image max : {spatial_elipse_initial.max()}")
    st.sidebar.caption(f"Image mean : {spatial_elipse_initial.mean()}")

    st.image(cv2.resize(spatial_elipse_initial,(512,512)))
    h,w = spatial_elipse_initial.shape
    spatial_elipse = np.absolute(AstroImageProcessing.apply_mean_mask_subtraction(spatial_elipse_initial,mean_filter))
    #footprint = disk(0.8)

    #spatial_elipse = AstroImageProcessing.apply_mean_mask_subtraction(spatial_elipse_initial,mean_filter)
    #spatial_elipse = rank.mean(spatial_elipse_initial, footprint=footprint)
    spatial_elipse = ((spatial_elipse-spatial_elipse.min())/(spatial_elipse.max() - spatial_elipse.min()))

    #psf = np.ones((5, 5)) / 25
    #spatial_elipse = restoration.richardson_lucy(spatial_elipse, psf, num_iter=30)

    print(spatial_elipse.max(),spatial_elipse.min(),spatial_elipse.mean())
    spatial_elipse = np.clip(spatial_elipse,min_value/65535,max_value/65535)
    spatial_elipse = ((spatial_elipse-spatial_elipse.min())/(spatial_elipse.max() - spatial_elipse.min()))
    st.image(cv2.resize(spatial_elipse,(512,512)))
    data = []
    for y in range(spatial_elipse.shape[0]):
        for x in range(spatial_elipse.shape[0]):
            if spatial_elipse[y, x] > spatial_elipse.mean():
                data.append([x, y,spatial_elipse[y,x]])
    print(len(data))

    kmeans = KMeans(n_clusters=2, random_state=0,n_init=3).fit(data*255)
    print(spatial_elipse.max())
    st.image(cv2.resize(spatial_elipse,(512,512)))

    for center in kmeans.cluster_centers_:
        (cx,cy)=(int(center[0]),int(center[1]))
        print(cx,cy)
        spatial_elipse = cv2.circle(spatial_elipse, (cx,cy), 5, (1), 1)
    
    print(spatial_elipse.max())
    st.image(cv2.resize(spatial_elipse,(512,512)))
else:
    st.header("Waiting for input file")