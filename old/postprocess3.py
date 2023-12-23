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
from scipy.ndimage import convolve

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

def perona_malik_filter(image, iterations, kappa, gamma=0.1):
    # Création des opérateurs de gradient
    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])

    for i in range(iterations):
        # Calcul des gradients
        ix = convolve(image, dx)
        iy = convolve(image, dy)

        # Calcul de la magnitude du gradient
        norm_grad = np.sqrt(ix**2 + iy**2)

        # Calcul du coefficient de diffusion
        c = np.exp(-(norm_grad / kappa)**2)

        # Application de la diffusion
        image = image + gamma * (convolve(ix, dx) + convolve(iy, dy)) * c

    return image

def erase_principal_disk(image, vector):

    # Exemple de profil d'intensité pour le disque central
    # Supposons que c'est un profil d'intensité linéaire pour simplifier l'exemple
    # Taille de l'image
    image_size = image.shape[0]

    # Créer une matrice pour le disque central
    disque_central = np.zeros((image_size, image_size))

    # Calculer la distance de chaque point au centre de l'image
    centre = image_size // 2
    for x in range(image_size):
        for y in range(image_size):
            distance = np.sqrt((x - centre)**2 + (y - centre)**2)
            distance_index = int(distance)

            # Utiliser le profil d'intensité pour définir la valeur du disque
            if distance_index < len(vector):
                disque_central[x, y] = vector[distance_index]

    # Afficher le disque central reconstruit
    #plt.imshow(disque_central, cmap='gray')
    #plt.title("Disque Central Reconstruit")
    #plt.colorbar()
    #plt.show()


print("--------------")
st.title('# Speckle interferometry analysis')
uploaded_file = st.file_uploader("Choose a file",type=['npy'])
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
level = st.sidebar.slider('Contour level', -1.0, 1.0, 0.0)
radius = st.sidebar.slider('radius', 1, 30, 1, 1)

if uploaded_file is not None:
    



    st.header(uploaded_file.name)
    st.divider()
    name = uploaded_file.name.split('.')[0]
    spatial_elipse_initial = load_image(uploaded_file)
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
    spatial_elipse = np.absolute(AstroImageProcessing.apply_mean_mask_subtraction(spatial_elipse_initial,mean_filter))
    #spatial_elipse =  (0.0+spatial_elipse_initial-spatial_elipse_initial.min())/(spatial_elipse_initial.max() - spatial_elipse_initial.min())
    #spatial_elipse = np.abs(spatial_elipse - ndimage.median_filter(spatial_elipse, mean_filter))
    spatial_elipse=spatial_elipse[1:h-1,1:w-1]*65535
    spatial_elipse=detect_and_remove_lines(spatial_elipse)


    spatial_elipse_filtered = (np.clip(spatial_elipse, min_value,max_value)-min_value)*(max_value-min_value)

    st.header("Correlation after mean substraction")
    # filtered_image = 
    show_image(perona_malik_filter(spatial_elipse_filtered, iterations=400, kappa=5),"After mean substraction",st)

    show_image(spatial_elipse_filtered,"After mean substraction",st)


    


    from skimage import color, data, restoration
    from scipy.signal import convolve2d
    psf = np.ones((5, 5)) / 25
    img = convolve2d(spatial_elipse_filtered*65535, psf, 'same')
    img += 0.1 * img.std() * np.random.standard_normal(img.shape)
    deconvolved_img,_ = restoration.unsupervised_wiener(img, psf)
    show_image(deconvolved_img,"wiener",st)

    image=(spatial_elipse_filtered/spatial_elipse_filtered.max() * 255).astype(np.uint8)
    image = cv2.bilateralFilter(image, 15, 75, 75) 

    image_bkp =spatial_elipse_filtered.copy()
    #cnt,fig = AstroImageProcessing.draw_contours(image, level)

    ret, thresh = cv2.threshold(image, image.mean()*0.5, 255, 0)
    contours, hierarchy = cv2.findContours((AstroImageProcessing.gaussian_sharpen(thresh,8,5)*255/thresh.max()).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours=sorted(contours, key=cv2.contourArea, reverse=True)


    #clustering = DBSCAN(eps=3, min_samples=2).fit(contours[0])
    
    """
    model = skl_cluster.SpectralClustering(n_clusters=3, affinity='nearest_neighbors', assign_labels='kmeans')
    labels = model.fit_predict(contours[0].squeeze(axis=1))
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    colors=[[255,127,0],[0,255,0],[0,0,255]]
    for (i,pt) in enumerate(contours[0].squeeze(axis=1)):
        color_image[pt[1],pt[0]]=colors[labels[i]]
        print(pt[1],pt[0],labels[i])

    pts = []
    for cnt in contours[0].squeeze(axis=1):
        pts.append([cnt[0],cnt[1]])

    kmeans = KMeans(n_clusters=20, random_state=0,n_init=100).fit(pts)
    for center in kmeans.cluster_centers_:
        (cx,cy)=(int(center[0]),int(center[1]))
        color_image = cv2.circle(color_image, (cx,cy), 1, (255,0,0), 1)


    st.image(cv2.resize(color_image,(512,512)))"""
    ellipse = cv2.fitEllipse(contours[0])
    centre, axes, angle = ellipse
    axe1, axe2 = axes[0] / 2, axes[1] / 2
    axe_majeur = max(axe1,axe2)
    axe_mineur = min(axe1,axe2)
    c = np.sqrt(axe_majeur**2 - axe_mineur**2)
    angle_rad = np.deg2rad(angle-90)


    #image2 = cv2.drawContours(image.copy(), contours,-1,255)
    image2 = cv2.ellipse(image.copy(),ellipse,255, 1)
    foyer1 = (int(centre[0] + c * np.cos(angle_rad)), int(centre[1] + c * np.sin(angle_rad)))
    foyer2 = (int(centre[0] - c * np.cos(angle_rad)), int(centre[1] - c * np.sin(angle_rad)))
    print(f"foyer {foyer1} {foyer2}")
    image2 = cv2.line(image2,foyer1, foyer2, 0,3)
    #ret,image = cv2.threshold((image/255).astype(np.uint8),int(min_value/255),255,cv2.THRESH_BINARY)
    st.header("Image after edge reshaping")
    st.caption("Contour detection")
    st.pyplot(fig)
    st.image(cv2.resize(image2,(512,512)))


    image=((image-image.min())/(image.max()-image.min()) * 255).astype(np.uint8)
    image = AstroImageProcessing.draw_circle(image)
    #image = cv2.circle(image, (image.shape[0]//2,image.shape[1]//2),radius, 0,-1)

    m = (foyer1[1]-foyer2[1])/(foyer1[0]-foyer2[0])
    mperp = -1/m
    b = image.shape[0]//2 *(1-mperp)

    #image=cv2.line(image,(0,int(b)),(image.shape[0],int(image.shape[0]*mperp+b)),0,radius)
    image=((image-image.min())/(image.max()-image.min()) * 255).astype(np.uint8)
    ret, thresh = cv2.threshold(image, image.mean(), 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours, key=cv2.contourArea, reverse=True)
    

    if len(contours)>1:
        contours = contours[0:2]

    #image = cv2.drawContours(image, contours,-1,(255))    
    
    for cnt in contours:
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centre = (cx, cy)
        #image = cv2.circle(image, (cx,cy),1,255,1)

    m = (foyer1[1]-foyer2[1])/(foyer1[0]-foyer2[0])*(1+level*5)
    b = image.shape[1]/2 *(1-m)
    
    d_max = math.sqrt(image.shape[0]**2 + image.shape[1]**2) / 2

    x_C = image.shape[0]/2
    y_C = image.shape[1]/2
    dx = 1 / math.sqrt(1 + m**2)  # Normalisé
    dy = m * dx

    curve = []
    image = spatial_elipse_filtered.copy()
    step = 1
    for d in range(0, int(d_max)*step):
        # Calculer les coordonnées des deux points possibles C2
        # Utilisation de la formule paramétrique de la droite
        x_C2_1 = x_C + d/step * dx
        y_C2_1 = y_C + d/step * dy

        x_C2_2 = x_C - d/step * dx
        y_C2_2 = y_C - d/step * dy

        if 0 <= x_C2_1 < image.shape[0] and 0 <= y_C2_1 < image.shape[1]:
            curve.append([d/step,image[int(y_C2_1),int(x_C2_1)]])

        
        if 0 <= x_C2_2 < image.shape[0] and 0 <= y_C2_2 < image.shape[1]:
            curve.append([-d/step,image[int(y_C2_2),int(x_C2_2)]])
    

    curve = sorted(curve, key=lambda pair: pair[0])

    """
    x_values =x
    y_values = y
    window_size = 7
      # window size must be odd
    poly_order = 3  # polynomial order
    if window_size > len(y_values):
        window_size = len(y_values) - 1 if len(y_values) % 2 == 0 else len(y_values)
    smoothed_y_values = savgol_filter(y_values, window_size, poly_order)


    #curve = [(x[i], smoothed_y_values[i]) for i in range(len(smoothed_y_values))]
"""
    maximum=0
    for i in range(len(curve)):
        print(i,curve[i][1])
        if curve[i][1] > maximum:
            maximum = curve[i][1]

    c=len(curve[:])//2
    
    print(f"max {maximum}")
    for i in range(-4,4):

        if curve[c+i][1]>0.9*maximum:
            curve[c+i][1]=maximum
    print("curve",curve[c-4][1],curve[c-3][1],curve[c-2][1],curve[c-1][1], curve[c][1], curve[c+1][1],curve[c+2][1])
    #y = y-smoothed_y_values
    x = [item[0] for item in curve]
    y = [item[1] for item in curve]

    grads = []
    x2=[]
    peak=[]
    last=0
    in_peak=False
    first_peak=-1
    for i in range(1,len(curve)-1):
        x2.append(i)
        grad = 0.0 + (curve[i+1][1]-curve[i][1])
        print(i, grad,curve[i][1],first_peak, in_peak)
        if grad>0:
            in_peak=True
            first_peak=-1
            
        if grad<0:
            if in_peak:
                if first_peak!=-1:
                    a="fp"
                    ind = i-(i - first_peak)//2
                else:
                    ind=i
                    a="in"
                    
                peak.append([ind,curve[ind][1]])
                print("+++"+a,ind, first_peak)
            first_peak = -1
            in_peak=False


        if grad==0 and in_peak and first_peak==-1:
            first_peak = i
        last = grad
        grads.append(grad)
    print("before sorted", peak)
    peak = sorted(peak, key=lambda pair: pair[1], reverse=True)
    print("after sorted", peak)
    if len(peak)>3:
        peak = peak[0:3]

    if len(peak)>=2:
        if len(peak)==3:
            dist1 = 0.099*(abs(curve[peak[1][0]][0]))
            dist2 = 0.099*(abs(curve[peak[2][0]][0]))
        else:
            dist = 0.099*(abs(curve[peak[1][0]][0]))
            dist1 = dist2 = dist
        st.header(f"Distance calculated : {dist1},{dist2}, {(dist1+dist2)/2}")

    #x = [item[0] for item in curve]
    #y = [item[1] for item in curve]


    #fig, ax = plt.subplots()
    #ax.plot(x_values,smoothed_y_values)
    #for i in peak:
    #    ax.axvline(x=curve[i[0]][0], color='r', linestyle='--')
    #st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(x,y)
    for i in peak:
        ax.axvline(x=curve[i[0]][0], color='r', linestyle='--')
    
    

    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    ax.plot(x2,grads)
    for i in peak:
        ax.axvline(x=i[0], color='r', linestyle='--')
    
    st.pyplot(fig)

    

    grad2=[]
    x2 = []
    for i in range(1,len(grads)):
        grad = grads[i]+0.0-grads[i-1]
        grad2.append(grad)
        x2.append(i)
    fig, ax = plt.subplots()
    ax.plot(x2,grad2)
    st.pyplot(fig)

    """
    print(m,b)
    maximumm = 0
    max_xy=[]

    curve = []
    for x in range(0,image.shape[0]):
        y = int(m * x + b)
        dist = math.sqrt((x - image.shape[0]/2)**2+(y-image.shape[1]/2)**2)
        if x < image.shape[0]//2:
            dist = -dist
        
        
        if y>0 and y<image.shape[1]:
            print(x,y, image.shape)
            curve.append([dist, image[y,x]])
            val = image[y,x]
            if val> maximumm:
                maximumm=val
                max_xy=[]

            if (val==maximumm):
                max_xy.append(x)

    x = np.mean(max_xy)
    y = int(m * x + b)
    dist = math.sqrt((image.shape[0]/2-x)**2+(image.shape[1]/2-y)**2)*0.099
    st.caption(f"Distance calculée : {dist}")
    print(curve)
    curve = sorted(curve, key=lambda pair: pair[0])
    x = [item[0] for item in curve]
    y = [item[1] for item in curve]
    fig, ax = plt.subplots()
    ax.plot(x,y)
    st.pyplot(fig)

#--------------
    for y in range(0,image.shape[1]):
        x = int((y-b)/m)
        dist = math.sqrt((x - image.shape[0]/2)**2+(y-image.shape[1]/2)**2)
        if y < image.shape[1]//2:
            dist = -dist
        
        
        if x>0 and x<image.shape[0]:
            print(x,y, image.shape)
            curve.append([dist, image[y,x]])
            val = image[y,x]
            if val> maximumm:
                maximumm=val
                max_xy=[]

            if (val==maximumm):
                max_xy.append(x)

    x = np.mean(max_xy)
    y = int(m * x + b)
    dist = math.sqrt((image.shape[0]/2-x)**2+(image.shape[1]/2-y)**2)*0.099
    st.caption(f"Distance calculée : {dist}")
    print(curve)
    curve = sorted(curve, key=lambda pair: pair[0])
    x = [item[0] for item in curve]
    y = [item[1] for item in curve]
    fig, ax = plt.subplots()
    ax.plot(x,y)
    st.pyplot(fig)
"""

    
    last = y[len(x)//2]
    dist_max = 0
    for i in range(1,len(x)//2):
        v1 = y[len(x)//2-i]
        v2 = y[len(x)//2-i]
        v = max(v1,v2)
        if v > last:
            dist_max=i
        last = v
    

    

    """
    for x in max_xy:
        image[int(m * x + b),x]=255
 
    image = cv2.circle(image,(int(x),int(y)),1,255,1)
    """

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