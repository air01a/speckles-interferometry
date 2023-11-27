import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import splitext, isfile, isdir, getsize
from imagemanager import ImageManager
from processing import AstroImageProcessing
from scipy.signal import correlate2d
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
import mpld3
from mpld3 import plugins
import streamlit.components.v1 as components
import imutils


def load_speckle_images(image_files):
    """ Charge les images de tavelures à partir d'une liste de chemins de fichiers. """
    imager = ImageManager()

    tab =  [imager.read_image(file) for file in image_files]
    i=0
    #for t in tab:
    #    cv2.imshow(f"test{i}",(t/255).astype(np.uint8))
    #    i+=1
    return tab

def autocorrelation(image):
    # Convertir l'image en niveaux de gris si elle est en couleur

    # Normaliser l'image
    image = image.astype(np.float32) - np.mean(image)
    image = image / np.std(image)

    # Calculer l'autocorrelation
    result = correlate2d(image, image, mode='full', boundary='fill', fillvalue=0)

    return (result.astype(np.uint16)*65535)


def align_images_using_cross_correlation(base_image, other_images):
    """ Aligner les images en utilisant la corrélation croisée. """
    aligned_images = [base_image]
    for img in other_images:
        # Calculer la corrélation croisée
        cc = cv2.matchTemplate(img, base_image, cv2.TM_CCORR)
        _, _, _, max_loc = cv2.minMaxLoc(cc)

        # Déterminer le décalage
        dx, dy = max_loc[0] - base_image.shape[1] // 2, max_loc[1] - base_image.shape[0] // 2

        # Appliquer le décalage
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_img = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))

        aligned_images.append(aligned_img)
    return aligned_images

def average_speckle_images(speckle_images):
    """ Calcule l'image moyenne à partir d'un ensemble d'images de tavelures. """
    sum = np.mean(speckle_images, axis=0)
    print(sum)
    return (sum*65535/sum.max()).astype(np.uint16)



def inverse_fourier_transform(fourier_image):
    # Appliquer la transformée de Fourier inverse
    f_ishift = np.fft.ifftshift(fourier_image)
    img_back = np.fft.ifft2(f_ishift)

    # Prendre la magnitude pour obtenir l'image réelle
    img_back = np.abs(img_back)

    return img_back

def process_speckle_interferometry(image_files):
    """ Traite un ensemble d'images de tavelures pour la speckle interferometry. """
    # Charger les images
    st.header("Sum of correlation")
    speckle_images = load_speckle_images(image_files)

    fouriers = []
    for im in speckle_images:
        fouriers.append(AstroImageProcessing.fourier_transform(im))
    
    # Aligner les images
    #aligned_images = align_images_using_cross_correlation(fouriers[0], fouriers[1:])

    # Calculer l'image moyenne
    average_image = average_speckle_images(fouriers) #average_speckle_images(aligned_images)

    # Appliquer la transformation de Fourier
    fourier_image = AstroImageProcessing.fourier_transform(average_image)
    fig = plt.figure(figsize=(8, 6))
    print(fourier_image.shape)
    #w = 190
    #h = w
    #x,y = (int(fourier_image.shape[1]/2-w/2),int(fourier_image.shape[0]/2-h/2))
    #print(x,y, w, h)
    # Afficher les résultats
    ax = fig.add_subplot()
    ax.imshow(autocorrelation(fourier_image), cmap = 'gray')
   # ax.imshow(autocorrelation(fourier_image)[y:int(y+y/2),x:x+w//2], cmap = 'gray')
    #ax2 = fig.add_subplot()
    #ax2.imshow(((autocorrelation(fourier_image))), cmap = 'gray')

    st.pyplot(fig)


def show_image_3d(image):

    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')


    # Utiliser la valeur des pixels comme coordonnées Z
    z = image

    # Afficher la surface
    ax.plot_surface(x, y, z, cmap='gray')

    # Définir les libellés des axes
    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Valeur des pixels')
    ax.view_init(56, 134, 0)
    # Afficher le graphique
    st.pyplot(fig)
    plt.show()

def to_st(image):
    return (image/255).astype(np.uint8)


def find_peak_intensity(image):
    # Convertir en niveaux de gris
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Trouver la position du pic d'intensité
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
    print(minVal, maxVal, minLoc, maxLoc)
    return maxLoc

def find_roi(image):
    st.header("Peak")
    col1, col2= st.columns(2)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    thresh = cv2.threshold(blurred, 10000, 65535, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    #with col1:
    #    st.image(thresh,clamp=True)

    contours = cv2.findContours(to_st(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    for c in cnts:
	# compute the center of the contour
        M = cv2.moments(c)
        print(M)
        l1 = c[:,:,0].max() - c[:,:,0].min()
        l2 = c[:,:,1].max() - c[:,:,1].min()
        l = int(max(l1,l2))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 2)

        cropped_image = image[cY-l:cY+l, cX-l:cX+l]
        print(cropped_image)
        print("contour")
        with col2:
            st.image(to_st(cropped_image),clamp=True)
    with col1:
        st.image(to_st(thresh),clamp=True)
    return cropped_image

    # Trouver le bounding box du plus grand contour
    #largest_contour = max(contours, key=cv2.contourArea)
    #x, y, w, h = cv2.boundingRect(largest_contour)
    #print("recadrage",x,y,w,h)
    # Recadrer l'image sur la zone lumineuse

    #cropped_image = image[y:y+h, x:x+w]
    #with col2:
    #    st.image(cropped_image,clamp=True)

# Afficher ou sauvegarder l'image recadré

    #fig, ax = plt.subplots()
    #ax.imshow(thresh, cmap="gray")
    #st.pyplot(ax)
    #plt.show()

def resize_with_padding(image, target_width, target_height):
    h, w = image.shape[:2]
    delta_w = target_width - w
    delta_h = target_height - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = 0
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def square_correl(image):
    st.header("Correlations")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("Square correlation")
        st.image(to_st(AstroImageProcessing.calculate_correlation(image, np.square(image))))
    with col2:
        st.caption("FFT")
        ft = AstroImageProcessing.fourier_transform(image)
        ft = (ft*65535/ft.max()).astype(np.uint16)
        st.image(to_st(ft))
    with col3:
        st.caption("FFT Auto correlation")
        show_image_3d(AstroImageProcessing.calculate_correlation(ft,ft))
    with col4:
        st.caption("FFT Square auto correlation")
        show_image_3d(AstroImageProcessing.calculate_correlation(ft,np.square(ft)))

    


st.title("Data exploration for binaries stars post processing")
path = "images/a258_f0006.fit"
imager = ImageManager()
image = imager.read_image(path)

col1, col2 = st.columns(2)
with col1:
    st.header("Original")
    st.image(to_st(image), clamp=True)
#image = AstroImageProcessing.levels(AstroImageProcessing.stretch(image,0.2),5000,0.1,65000)
with col2:
    st.header("Processed")
    st.image(to_st(image),clamp=True)
st.header("3D View of image")
image = find_roi(image)

show_image_3d(image)
#maxloc = find_peak_intensity(image)
#print(maxloc)
square_correl(image)

image_files = []
dir = "images"
for file in listdir(dir):
    file_path = dir + '/' + file
    if isfile(file_path) and splitext(file_path)[1]=='.fit':
        image_files.append(file_path)
    
# Liste des chemins de fichiers des images de tavelures
#image_files = ['images/speckle1.jpg', 'images/speckle2.jpg', 'images/speckle3.jpg'] # Remplacez par vos fichiers

# Traitement des images de speckle
#process_speckle_interferometry(image_files)
