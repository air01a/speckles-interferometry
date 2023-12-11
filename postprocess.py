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

import imagepers


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


print("--------------")
st.title('# Speckle interferometry analysis')
uploaded_file = st.file_uploader("Choose a file",type=['npy'])
if uploaded_file is not None:
    if 'previous_file' in st.session_state and uploaded_file.name!=st.session_state['previous_file'] :
        print("rest")
        st.empty()
    st.session_state['previous_file'] = uploaded_file.name


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
    print(histogram)
    fig, ax = plt.subplots()

    ax.set_title('Histogramme de l\'image')
    ax.set_xlabel('Intensité des pixels')
    ax.set_ylabel('Nombre de pixels')
    ax.bar(bin_edges[:-1], histogram, width=size, edgecolor='black')
    st.pyplot(fig)
    

    h,w = spatial_elipse_initial.shape
    spatial_elipse = np.absolute(AstroImageProcessing.apply_mean_mask_subtraction(spatial_elipse_initial,mean_filter))
    spatial_elipse=spatial_elipse[1:h-1,1:w-1]*65535
    spatial_elipse=detect_and_remove_lines(spatial_elipse)


    spatial_elipse_filtered = (np.clip(spatial_elipse, min_value,max_value)-min_value)*(max_value-min_value)
    st.header("Correlation after mean substraction")
    show_image(spatial_elipse_filtered,"After mean substraction",st)




    image=(spatial_elipse_filtered/spatial_elipse_filtered.max() * 255).astype(np.uint8)
    image_bkp =spatial_elipse_filtered.copy()
    cnt,fig = AstroImageProcessing.draw_contours(image, level)

    ret, thresh = cv2.threshold(image, image.mean()*0.5, 255, 0)
    contours, hierarchy = cv2.findContours((AstroImageProcessing.gaussian_sharpen(thresh,8,5)*255/thresh.max()).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipse = cv2.fitEllipse(contours[0])
    centre, axes, angle = ellipse
    axe1, axe2 = axes[0] / 2, axes[1] / 2
    axe_majeur = max(axe1,axe2)
    axe_mineur = min(axe1,axe2)
    c = np.sqrt(axe_majeur**2 - axe_mineur**2)
    angle_rad = np.deg2rad(angle-90)



    foyer1 = (int(centre[0] + c * np.cos(angle_rad)), int(centre[1] + c * np.sin(angle_rad)))
    foyer2 = (int(centre[0] - c * np.cos(angle_rad)), int(centre[1] - c * np.sin(angle_rad)))
    #ret,image = cv2.threshold((image/255).astype(np.uint8),int(min_value/255),255,cv2.THRESH_BINARY)
    st.header("Image after edge reshaping")
    st.caption("Contour detection")
    st.pyplot(fig)


    image=(image/image.max() * 255).astype(np.uint8)
    image = AstroImageProcessing.draw_circle(image)
    #image = cv2.circle(image, (image.shape[0]//2,image.shape[1]//2),radius, 0,-1)

    m = (foyer1[1]-foyer2[1])/(foyer1[0]-foyer2[0])
    mperp = -1/m
    b = image.shape[0]//2 *(1-mperp)

    image=cv2.line(image,(0,int(b)),(image.shape[0],int(image.shape[0]*mperp+b)),0,radius)
    show_image(image,"Circled",st,"gray")
    show_image_3d(image,st)
    



    if st.sidebar.button("Save", type="primary"):
        output = (image/image.max() * 65535).astype(np.uint16)
        cv2.imwrite(f"results/{name}_output.png", cv2.resize(output,(512,512)))
        dic = {'radius':radius,'axe1':axe1,'axe2':axe2,'max_filter':max_value,'min_filter':min_value, "n_filter":mean_filter,"min":int(spatial_elipse_initial.min()*65535),"max":int(65535*spatial_elipse_initial.max()),
               "mean":int(65535*spatial_elipse_initial.mean()),'histogram':histogram.tolist()}
        with open(f"results/{name}.json", "w") as outfile:
            json.dump(dic, outfile)
else:
    st.header("Waiting for input file")