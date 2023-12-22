import numpy as np
import cv2
from matplotlib import pyplot as plt

import streamlit as st
from skimage.draw import ellipse
import math
from processing import AstroImageProcessing
from pyplot_utils import show_image, show_image_3d
import json
from scipy.ndimage import convolve


from imagecalculation import calculate_line, calculate_perp, detect_and_remove_lines, erase_principal_disk, \
                             find_contours,find_ellipse,get_vector_from_direction, find_peaks, slice_from_direction

@st.cache_data
def load_image(file):
    image = np.load(file)
    spatial_elipse = image/image.max()
    return spatial_elipse


#####################################################
# Main streamlit definition
#####################################################
print("--------------")
st.title('# Speckle interferometry analysis')

#### Form input fields
uploaded_file = st.file_uploader("Choose a file",type=['npy'])
if uploaded_file is not None:
    if 'previous_file' in st.session_state and uploaded_file.name!=st.session_state['previous_file'] :
        st.empty()
    st.session_state['previous_file'] = uploaded_file.name


st.sidebar.header("Input Parameters")
max_filter_value = st.sidebar.slider('Max filtering', 300,65535, 300, )
min_filter_value = st.sidebar.slider('Min filtering', 400,12000, 400, )

max_value = st.sidebar.slider('Max filtering', 0,max_filter_value, 300, )
min_value = st.sidebar.slider('Min filtering', 0, min_filter_value, 0)
mean_filter = st.sidebar.slider('Mean filtering', 3, 14, 3, 2)
level = st.sidebar.slider('Contour level', -1.0, 1.0, 0.0)
radius = st.sidebar.slider('radius', 1, 30, 1, 1)


#### When file has been uploaded
if uploaded_file is not None:
    st.header(uploaded_file.name)
    st.divider()
    name = uploaded_file.name.split('.')[0]
    #### Load and display image
    spatial_elipse_initial = load_image(uploaded_file)
    show_image(spatial_elipse_initial,'Initial',st)
    h,w = spatial_elipse_initial.shape

    st.sidebar.caption(f"Filter Max : {max_value}")
    st.sidebar.caption(f"Filter Min : {min_value}")
    st.sidebar.caption(f"Mean filter : {mean_filter}")
    st.sidebar.caption(f"Contour level : {level}")
    st.sidebar.caption(f"Image min : {spatial_elipse_initial.min()}")
    st.sidebar.caption(f"Image max : {spatial_elipse_initial.max()}")
    st.sidebar.caption(f"Image mean : {spatial_elipse_initial.mean()}")
    st.sidebar.caption(f"radius : {radius}")

    #### Calculate and display histogram
    st.header("Histogramme")
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
    

    #### Apply mean filter substraction to remove noise
    h,w = spatial_elipse_initial.shape
    spatial_elipse = np.absolute(AstroImageProcessing.apply_mean_mask_subtraction(spatial_elipse_initial,mean_filter))
    spatial_elipse=spatial_elipse[1:h-1,1:w-1]*65535
    spatial_elipse=detect_and_remove_lines(spatial_elipse)

    #### Clip image according to filters inputs
    spatial_elipse_filtered = (np.clip(spatial_elipse, min_value,max_value)-min_value)*(max_value-min_value)

    st.header("Correlation after mean substraction")
    show_image(spatial_elipse_filtered,"After mean substraction",st)


    #### Find best ellipse to fit
    contours = find_contours(spatial_elipse_filtered)
    (ellipse, major_axe, minor_axe, angle_rad, focus1, focus2) = find_ellipse(contours[0])
    image2 = cv2.ellipse(spatial_elipse_filtered.copy(),ellipse,65535, 1)
    st.header("Calcul de l'ellipse la plus proche")
    show_image(image2, "Ellipse", st)


    #### Calculate perpendicular line from the ellipse main axis
    x_C = spatial_elipse.shape[0]/2
    y_C = spatial_elipse.shape[1]/2
    (m,p) = calculate_line((focus1[0],focus1[1]),(focus2[0],focus2[1]))
    (m_perp,p_perp) = calculate_perp(m, (spatial_elipse.shape[0]/2, spatial_elipse.shape[1]/2))

    #### Get image values along this line
    vec = get_vector_from_direction(spatial_elipse,m_perp,(x_C,y_C),3*int(minor_axe))

    #### Use these values to extrapole central peak contribution and remove it from the image
    erased_image,central_disk = erase_principal_disk(spatial_elipse, vec)
    mean = erased_image.mean()
    erased_image = (np.clip(erased_image, mean,65535))
    erased_image = (erased_image-erased_image.min())*(erased_image.max()-erased_image.min())*65535
    
    #### Display results
    st.header("Calcul de la contribution du pic central")
    st.write(central_disk)
    show_image(central_disk,"central disk",st)
    cv2.circle(erased_image, (int(x_C),int(y_C)), radius, 0,-1)
    st.header("Image sans le pic central")
    show_image(erased_image,"erasing",st)

    #### Slice the image along the elipse main axe
    curve = slice_from_direction(erased_image,m, (x_C, y_C))

    #### Find peaks using gradients
    peaks,grads = find_peaks(curve)


    if len(peaks)>=2:
        peaks = peaks[0:2]
        dist1 = 0.099*(abs(curve[peaks[1][0]][0]))
        dist2 = 0.099*(abs(curve[peaks[0][0]][0]))
        st.header(f"Distance calculated : {dist1},{dist2}, {(dist1+dist2)/2}")

    #### Find peaks using max values along x and -x axis

    i=0
    left_max={}
    right_max={}
    while curve[i][0]<0:
        left_max[curve[i][1]]=curve[i][0]
        right_max[curve[i+len(curve)//2][1]]=curve[i+len(curve)//2][0]
        i+=1
    
    lm = sorted(left_max.keys(), reverse=True)
    lr = sorted(right_max.keys(), reverse=True)
    print(left_max[lm[0]], right_max[lr[0]])
    print(left_max[lm[1]], right_max[lr[1]])
    print(left_max[lm[2]], right_max[lr[2]])
    st.header(f"Distance calculated : {left_max[lm[0]]},{right_max[lr[0]]}, {0.099*(abs(left_max[lm[0]])+abs(right_max[lr[0]]))/2}")


    #### Display

    st.header("Analyse des pics dans la direction principale de l'ellipse")
    x = [item[0] for item in curve]
    y = [item[1] for item in curve]
    fig, ax = plt.subplots()
    ax.plot(x,y)
    for i in peaks:
        ax.axvline(x=curve[i[0]][0], color='r', linestyle='--')
    st.pyplot(fig)
    


    fig, ax = plt.subplots()
    ax.plot(x[1:-1],grads)
    for i in peaks:
        ax.axvline(x=i[0], color='r', linestyle='--')
    
    st.pyplot(fig)

    




    if st.sidebar.button("Save", type="primary"):
        output = (spatial_elipse_filtered * 65535).astype(np.uint16)
        cv2.imwrite(f"results/{name}_output.png", cv2.resize(output,(512,512)))
        dic = {'radius':radius,'axe1':major_axe,'axe2':minor_axe,'max_filter':max_value,'min_filter':min_value, "n_filter":mean_filter,"min":int(spatial_elipse_initial.min()*65535),"max":int(65535*spatial_elipse_initial.max()),
               "mean":int(65535*spatial_elipse_initial.mean()),'histogram':histogram.tolist()}
        with open(f"results/{name}.json", "w") as outfile:
            json.dump(dic, outfile)
else:
    st.header("Waiting for input file")