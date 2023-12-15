import numpy as np
import cv2
from matplotlib import pyplot as plt

import streamlit as st
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
import math
from skimage.draw import line_aa
from processing import AstroImageProcessing
from pyplot_utils import show_image, show_image_3d
import json
from sklearn.cluster import DBSCAN
from scipy import ndimage
import imagepers
from sklearn.cluster import KMeans
import sklearn.cluster as skl_cluster
import joblib
from scipy.signal import find_peaks

from scipy.signal import savgol_filter
def detect_and_remove_lines(image):

    (h,w) = image.shape
    for i in range(int(w/2-0.3*w),int(w/2+0.3*w)):
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
    return spatial_elipse

@st.cache_data
def load_model():
    model_radius = joblib.load('model_radius.pkl')
    model_n_filter = joblib.load('model_nfilter.pkl')
    model_min = joblib.load('model_min_filter.pkl')
    model_max = joblib.load('model_max_filter.pkl')
    return(model_min, model_max, model_n_filter, model_radius)

def calculer_cercle_circonscrit(pts):
    x1, y1 = pts[0,0]
    x2, y2 = pts[1,0]
    x3, y3 = pts[2,0]

    # Calcul des médiatrices pour les segments [P1P2] et [P1P3]
    ma = (y2 - y1) / (x2 - x1)
    mb = (y3 - y1) / (x3 - x1)

    # Calcul du centre du cercle (xc, yc)
    xc = (ma * mb * (y1 - y3) + mb * (x1 + x2) - ma * (x1 + x3)) / (2 * (mb - ma))
    yc = -1/ma * (xc - (x1 + x2) / 2) + (y1 + y2) / 2

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

def set_shape_pixels_to_zero(img):
    # Read the image in grayscale

    # Threshold the image to make sure we have a binary image (0 or 255)
    _, binary_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    # Find all connected components (also known as blobs or shapes in the image)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8, ltype=cv2.CV_32S)

    # Assuming the central shape is the one with the centroid closest to the center of the image
    image_center = np.array([img.shape[1]/2, img.shape[0]/2])
    distances = np.sqrt((centroids[:,0] - image_center[0]) ** 2 + (centroids[:,1] - image_center[1]) ** 2)
    central_shape_label = np.argmin(distances[1:]) + 1  # +1 to avoid the background label

    # Set the pixels of the central shape to zero
    processed_img = np.where(labels == central_shape_label, 0, img).astype(np.uint8)

    # Save the processed image or return it
    # For this example, let's just return the NumPy array of the processed image
    return processed_img


print("--------------")
st.title('# Speckle interferometry analysis')
uploaded_file = st.file_uploader("Choose a file",type=['npy'])
spatial_elipse_initial = load_image('results/cou619.npy')



if uploaded_file is not None:
    if 'previous_file' in st.session_state and uploaded_file.name!=st.session_state['previous_file'] :
        st.empty()
    st.session_state['previous_file'] = uploaded_file.name
    
#(model_min, model_max, model_n_filter, model_radius) = load_model()
st.sidebar.header("Input Parameters")
max_filter_value = st.sidebar.slider('Max filtering', 300,65535, 300, )
min_filter_value = st.sidebar.slider('Min filtering', 400,12000, 400, )

max_value = st.sidebar.slider('Max filtering', 0,max_filter_value, 300, )
min_value = st.sidebar.slider('Min filtering', 0, min_filter_value, 0)
mean_filter = st.sidebar.slider('Mean filtering', 3, 14, 3, 2)
level = st.sidebar.slider('Contour level', 0.0, 1.0, 0.8, 0.1)
radius = st.sidebar.slider('radius', 1, 30, 1, 1)

if uploaded_file is not None:
    



    st.header(uploaded_file.name)
    st.divider()
    name = uploaded_file.name.split('.')[0]
    spatial_elipse_initial = load_image(uploaded_file)


    h,w = spatial_elipse_initial.shape
    spatial_elipse =  (0.0+spatial_elipse_initial-spatial_elipse_initial.min())/(spatial_elipse_initial.max() - spatial_elipse_initial.min())
    spatial_elipse = ndimage.prewitt(spatial_elipse)#np.abs(spatial_elipse - ndimage.median_filter(spatial_elipse, 21))
    spatial_elipse=spatial_elipse[1:h-1,1:w-1]*65535
    spatial_elipse=detect_and_remove_lines(spatial_elipse)
    show_image(spatial_elipse)

    #spatial_elipse_initial[spatial_elipse_initial.shape[0]//2,spatial_elipse_initial.shape[1]//2]=0
    
    #spatial_elipse_initial[spatial_elipse_initial.shape[0]//2,spatial_elipse_initial.shape[1]//2]=spatial_elipse_initial.max()
    h,w = spatial_elipse_initial.shape
    st.caption(int(spatial_elipse_initial.min()*65535))
    st.sidebar.caption(f"Filter Max : {max_value}")
    st.sidebar.caption(f"Filter Min : {min_value}")
    st.sidebar.caption(f"Mean filter : {mean_filter}")
    st.sidebar.caption(f"Contour level : {level}")
    st.sidebar.caption(f"Image min : {spatial_elipse_initial.min()}")
    st.sidebar.caption(f"Image max : {spatial_elipse_initial.max()}")
    st.sidebar.caption(f"Image mean : {spatial_elipse_initial.mean()}")
    st.sidebar.caption(f"radius : {radius}")

    size=5
    flattened_image = (spatial_elipse_initial*255).flatten()
    bins = np.arange(0, 256, size)
    histogram, bin_edges = np.histogram(flattened_image, bins)
    fig, ax = plt.subplots()

    ax.set_title('Histogramme de l\'image')
    ax.set_xlabel('Intensité des pixels')
    ax.set_ylabel('Nombre de pixels')
    ax.bar(bin_edges[:-1], histogram, width=size, edgecolor='black')
    st.pyplot(fig)
    


    h,w = spatial_elipse_initial.shape
    #spatial_elipse = np.absolute(AstroImageProcessing.apply_mean_mask_subtraction(spatial_elipse_initial,mean_filter))
    spatial_elipse =  (0.0+spatial_elipse_initial-spatial_elipse_initial.min())/(spatial_elipse_initial.max() - spatial_elipse_initial.min())
    spatial_elipse = np.abs(spatial_elipse - ndimage.median_filter(spatial_elipse, mean_filter))
    spatial_elipse=spatial_elipse[1:h-1,1:w-1]*65535
    spatial_elipse=detect_and_remove_lines(spatial_elipse)


    spatial_elipse_filtered = (np.clip(spatial_elipse, min_value,max_value)-min_value)*(max_value-min_value)
    st.header("Correlation after mean substraction")
    show_image(spatial_elipse_filtered,"After mean substraction",st)



    thresh = spatial_elipse_filtered.copy()
    #ret, thresh = cv2.threshold(image, image.mean()*0.5, 255, 0)
    #thresh = set_shape_pixels_to_zero(thresh)
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if thresh[i,j]!=0:
                print(thresh[i,j])



    coordinates = np.argwhere(thresh > 0.01)
    print(coordinates)
    center = np.array([thresh.shape[1]/2, thresh.shape[0]/2])
    print(center)
    dist = math.sqrt((coordinates[0,0]-coordinates[1,0])**2 + (coordinates[0,1]-coordinates[1,1])**2)
    print(f"distance : {dist/2 * 0.099}")
    thresh[thresh.shape[1]//2, thresh.shape[0]//2]=thresh.max()
    show_image(thresh, "Test",st)
else:
    st.header("Waiting for input file")