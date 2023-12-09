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





print("--------------")
st.title('# Speckle interferometry analysis')

st.sidebar.header("Input Parameters")
max_filter_value = st.sidebar.slider('Max filtering', 300,65535, 300, )
uploaded_file = st.file_uploader("Choose a file",type=['npy'])
max_value = st.sidebar.slider('Max filtering', 0,max_filter_value, 300, )
min_value = st.sidebar.slider('Min filtering', 0, 1200, 0)
mean_filter = st.sidebar.slider('Mean filtering', 3, 14, 3, 2)
level = st.sidebar.slider('Contour level', 0.0, 1.0, 0.8, 0.1)

if uploaded_file is not None:
    st.header(uploaded_file.name)
    st.divider()
    name = uploaded_file.name.split('.')[0]
    spatial_elipse_initial = load_image(uploaded_file)
    h,w = spatial_elipse_initial.shape
    
    st.sidebar.caption(f"Filter Max : {max_value}")
    st.sidebar.caption(f"Filter Min : {min_value}")
    st.sidebar.caption(f"Mean filter : {mean_filter}")
    st.sidebar.caption(f"Contour level : {level}")
    st.sidebar.caption(f"Image min : {spatial_elipse_initial.min()}")
    st.sidebar.caption(f"Image max : {spatial_elipse_initial.max()}")
    st.sidebar.caption(f"Image mean : {spatial_elipse_initial.mean()}")


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
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    show_image(image,"Circled",st,"gray")

    show_image_3d(image,st)
    st.caption(f"Contour numbers {len(cnt)}")
    

    g0 = imagepers.persistence(spatial_elipse_initial)
    print("g0",g0)

    g1 = imagepers.persistence(image_bkp)
    print("g1",len(g1))
    print(g1)
    im = np.zeros(image_bkp.shape)
    im=cv2.circle(im,(image_bkp.shape[0]//2,image_bkp.shape[0]//2),1,127,1)
    im = cv2.ellipse(im,ellipse,255,1)
    x1 = foyer1[0]
    y1 = foyer1[1]
    m = (y1-centre[1])/(x1-centre[0])
    p = y1-m*x1
    for pt in g1:
        coord = pt[0]
        y0 = int(m*coord[0]+p)
        if coord[1]!=0:
            print(abs(y0/coord[1]-1))
        if coord[1]!=0 and abs(y0/coord[1]-1)<0.01:
            im[coord[1],coord[0]] =127
    show_image(im,"homo",st)




    if st.sidebar.button("Save", type="primary"):
        output = (image/image.max() * 65535).astype(np.uint16)
        cv2.imwrite(f"results/{name}_output.png", cv2.resize(output,(512,512)))
        dic = {'max_filter':max_value,'min_filter':min_value, "n_filter":mean_filter,"min":spatial_elipse_initial.mean(),"max":spatial_elipse_initial.max(),
               "mean":spatial_elipse_initial.mean()}
        with open(f"results/{name}.json", "w") as outfile:
            json.dump(dic, outfile)
else:
    st.header("Waiting for input file")