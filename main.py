import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import splitext, isfile
from imagemanager import ImageManager
from processing import AstroImageProcessing

import streamlit as st


import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

def load_speckle_images(image_files):
    """ Charge les images de tavelures à partir d'une liste de chemins de fichiers. """
    imager = ImageManager()
    im_per_column = 6
    tab =  [roi for roi in [AstroImageProcessing.find_roi(imager.read_image(file)) for file in image_files] if roi is not None]

    max_w = 0
    max_h = 0
    for im in tab:
        (h, w) = im.shape
        max_w = max(w, max_w)
        max_h = max(h, max_h)
    columns = st.columns(im_per_column)
    for index, im in enumerate(tab):
        tab[index] = AstroImageProcessing.resize_with_padding(im, max_w, max_h)
        with columns[index % im_per_column]:
            st.image(AstroImageProcessing.to_int8(tab[index]))
    return tab

def psdDeconvolveLPF(psdBinary, psdReference, lpfRadius=None):


    imgSize = np.shape(psdReference)[1] # Calculate dimension of image
    imgCenter = int(imgSize/2)          # Center index of image

    # Create centered meshgrid of image
    xx,yy = np.meshgrid(np.arange(imgSize),np.arange(imgSize))
    xx = np.subtract(xx,imgCenter)
    yy = np.subtract(yy,imgCenter)
    rr = np.power(np.power(xx,2)+np.power(yy,2),0.5)
    
    # Create LPF filter image if specified
    if (lpfRadius != None):   
        lpf = np.exp(-(np.power(rr,2)/(2*np.power(lpfRadius,2))))
    else: 
        lpf = np.zeros((imgSize,imgSize))
        lpf.fill(1)
    print(lpf.shape)
    print(psdBinary.shape)
    # Perform wiener filtering
    return psdBinary*(1/psdReference)*lpf       

def apply_mean_mask_subtraction(image, k_size=3):
    """
    Applique une soustraction de masque moyen à l'image avec des noyaux croissants.

    :param image: Image d'entrée (autocorrelogramme).
    :param max_kernel_size: Taille maximale du noyau carré (doit être impair).
    :return: Image traitée.
    """
    processed_image = image.copy()
    # Créer un noyau moyen
    kernel = np.ones((k_size, k_size), np.float32) / (k_size * k_size)

    # Appliquer le filtre moyenneur
    mean_filtered = np.abs(cv2.filter2D(image, -1, kernel))
    #mean_filtered = (mean_filtered - mean_filtered.min()) / (mean_filtered.max() - mean_filtered.min())
    # Soustraire le résultat filtré de l'image originale
    #processed_image = cv2.subtract(processed_image, mean_filtered)

    return mean_filtered

def filter(image):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    radius = 30  # Radius of the low-pass filter
    mask = np.zeros((rows, cols), np.uint8)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
    mask[mask_area] = 1


    # Appliquer le masque
    f_shift_filtered = image * mask
    return f_shift_filtered


def align2_images(images):
    # Trouver les coordonnées des pixels les plus lumineux dans chaque image
    max_coords = [np.unravel_index(np.argmax(img), img.shape) for img in images]

    # Calculer le décalage nécessaire pour chaque image
    max_x, max_y = max(coords[0] for coords in max_coords), max(coords[1] for coords in max_coords)
    offsets = [(max_x - x, max_y - y) for x, y in max_coords]

    # Agrandir les images et appliquer le décalage
    new_size = (images[0].shape[0] + 2*max_x, images[0].shape[1] + 2*max_y)
    aligned_images = []
    for img, offset in zip(images, offsets):
        new_img = np.zeros(new_size)
        new_img[offset[0]:offset[0]+img.shape[0], offset[1]:offset[1]+img.shape[1]] = img
        aligned_images.append(new_img)

    return aligned_images

def isolate_peaks(image, sigma=2, min_distance=5, num_peaks=10):
    """
    Isoler les trois principaux pics dans une image 2D.

    :param image: Image d'entrée 2D.
    :param sigma: Écart-type pour le filtre gaussien.
    :param min_distance: Distance minimale entre les pics détectés.
    :param num_peaks: Nombre de pics à isoler.
    :return: Image avec seulement les trois principaux pics.
    """
    # Appliquer un filtre gaussien pour réduire le bruit
    smoothed_image = gaussian_filter(image, sigma=sigma)
    #smoothed_image = ndi.maximum_filter(image, size=20, mode='constant')

    # Détection des maxima locaux
    coordinates = peak_local_max(smoothed_image, min_distance=min_distance, num_peaks=num_peaks)

    # Créer une nouvelle image pour afficher les pics
    peak_image = np.zeros_like(image)
    for coord in coordinates:
        peak_image[coord[0], coord[1]] = image[coord[0], coord[1]]

    return peak_image


def process_speckle_interferometry(image_files):
    """ Traite un ensemble d'images de tavelures pour la speckle interferometry. """
    # Charger les images

    speckle_images = load_speckle_images(image_files)

    fouriers = []
    psd = []
    for im in speckle_images:
        fouriers.append(AstroImageProcessing.fourier_transform(im))
        psd.append(np.square(np.abs(AstroImageProcessing.fourier_transform(im))))
    
    psd_average= AstroImageProcessing.average_images((psd))
    spatial = AstroImageProcessing.inverse_fourier_transform((psd_average))
    spatial = spatial.real

    
    plt.figure()
    plt.imshow(spatial)
    plt.title("Observed and expected secondary locations")
    plt.show()    
  
  
    print(spatial.dtype)
    np.save("spatial.npy",spatial)
    show_image_3d(spatial)

    # Capture ellipse
    spatial_elipse = spatial/spatial.max()
    
    #spatial_elipse = spatial_elipse/spatial_elipse.max()
    spatial_elipse = np.absolute(apply_mean_mask_subtraction(spatial_elipse,3))
    show_image_3d(spatial_elipse)
    for i in range(3):
        spatial_elipse = np.clip(spatial_elipse, spatial_elipse.mean(),spatial_elipse.mean()*5)
        spatial_elipse = np.clip(spatial_elipse, spatial_elipse.mean()/2,spatial_elipse.max())
    show_image_3d(spatial_elipse)
    
    plt.figure()
    plt.imshow(spatial_elipse)
    plt.title("Observed and expected secondary locations")
    plt.show()   
    spatial_elipse = (spatial_elipse/spatial.max() * 65535).astype(np.uint16)
    cv2.imwrite('result_elipse.png', spatial_elipse)
    
    #spatial[spatial < spatial.mean()*5] = 0

    #spatial+=spatial.min()
    #spatial = np.clip(spatial,0, 1e10)
    show_image_3d(spatial)
    plt.figure()
    plt.imshow(spatial)
    plt.title("Observed and expected secondary locations")
    plt.show()

    spatial = (spatial/spatial.max() * 65535).astype(np.uint16)
    cv2.imwrite('result.png', spatial)
    #AstroImageProcessing.draw_contours(spatial)
    """
    sum_images = AstroImageProcessing.sum_images(speckle_images)

    
    st.header("Square correlation")
    st.caption("Sum images spatial")
    st.image(cv2.resize(AstroImageProcessing.to_int8(sum_images),(500,500)))



    fourier = (AstroImageProcessing.fourier_transform(sum_images))
    fourier2 = np.absolute(fourier)
    fourier2 = AstroImageProcessing.to_int8(fourier2)
    st.caption("SUM fourier transform")
    st.image(cv2.resize(fourier2,(500,500)))

    auto_correlation = np.absolute(AstroImageProcessing.crosscorfft(fourier, fourier))
    #auto_correlation = (auto_correlation*255/auto_correlation.max()).astype(np.uint8)
    st.caption("Auto corelation sum(fft)")
    st.image(AstroImageProcessing.to_int8(cv2.resize(auto_correlation,(500,500))))
    fouriers = []
    for im in speckle_images:
        fouriers.append(AstroImageProcessing.fourier_transform(im))

    # Calculer l'image moyenne
    average_fouriers = np.absolute(AstroImageProcessing.sum_images(fouriers)) #average_speckle_images(aligned_images)

    # Appliquer la transformation de Fourier
    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot()
    st.caption("AUTOCORRELATION(FFT(SUM(Images)))")
    ax.imshow(np.absolute(AstroImageProcessing.crosscorfft(average_fouriers,average_fouriers)))
    st.pyplot(fig)
    image = np.absolute(AstroImageProcessing.fourier_transform(np.square(AstroImageProcessing.fourier_transform(sum_images))))
    image = image
    st.caption("FFT(FFT²)")
    ax.imshow((image))
    st.pyplot(fig)
    result=[]
    for i in speckle_images:

        tmp = AstroImageProcessing.crosscorr(i,i)
        st.image(AstroImageProcessing.to_int8(tmp))
        result.append(tmp)

    st.image(cv2.resize(AstroImageProcessing.to_int8(sum_images),(512,512)))


    reference_location = find_brightest_pixel(speckle_images[0])
    print(reference_location)
    aligned_images = align_images(speckle_images, reference_location)
    sum_images2 = AstroImageProcessing.sum_images(aligned_images)
    ax = fig.add_subplot()

    ax.imshow(AstroImageProcessing.levels(sum_images2,13000,1,65535))
    #plt.show()
    #show_image_3d(sum_images)
    #show_image_3d(sum_images2)

    for i in range(0,9):
        fig = plt.figure(figsize=(8, 6))
        image=apply_mean_mask_subtraction(sum_images2   ,3+i*2)
        ax = fig.add_subplot()
        ax.imshow(image)
        #plt.show()
    # Calcul de la valeur moyenne des pixels
    mean_value = np.mean(sum_images2)

    # Création d'un ensemble de données avec les coordonnées x, y et la valeur du pixel
    # Uniquement pour les pixels dont la valeur dépasse la moyenne
    data = []
    for y in range(sum_images2.shape[0]):
        for x in range(sum_images2.shape[1]):
            if sum_images2[y, x] > mean_value:
                data.append([x, y, sum_images2[y, x]])

    data = np.array(data)

    # Application de l'algorithme K-means avec 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    cluster_centers = kmeans.cluster_centers_
    clustered_image = np.copy(image)
    for i in range(sum_images2.shape[0]):
        for j in range(sum_images2.shape[1]):
            if sum_images2[i, j] > mean_value:
                X = [j,i,sum_images2[i, j]]
                print(X)
                print(kmeans.predict([X]))

    # Afficher l'image originale et l'image après clustering
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Image Originale")
    plt.subplot(1, 2, 2)
    plt.imshow(clustered_image, cmap='gray')
    plt.title("Image Après Clustering")
    plt.show()
"""
    
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
    st.pyplot(fig)
    plt.show()

def square_correl(image):
    st.header("Correlations")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("Square correlation")
        st.image(AstroImageProcessing.to_int8(AstroImageProcessing.calculate_correlation(image, np.square(image))))
    with col2:
        st.caption("FFT")
        ft = AstroImageProcessing.fourier_transform(image).real()
        ft = (ft*65535/ft.max()).astype(np.uint16)
        st.image(AstroImageProcessing.to_int8(ft))
    with col3:
        st.caption("FFT Auto correlation")
        show_image_3d(AstroImageProcessing.calculate_correlation(ft,ft))
    with col4:
        st.caption("FFT Square auto correlation")
        show_image_3d(AstroImageProcessing.calculate_correlation(ft,np.square(ft)))


def find_brightest_pixel(image):
    """ Trouver les coordonnées du pixel le plus brillant dans l'image. """
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(image)
    return maxLoc

def align_images(images, reference_location):
    """ Aligner toutes les images par rapport à un emplacement de référence. """
    aligned_images = []
    print(images[0].shape)
    h, w = images[0].shape

    for img in images:
        # Trouver le pixel le plus brillant
        brightest_pixel = find_brightest_pixel(img)

        # Calculer le décalage nécessaire
        dx = reference_location[0] - brightest_pixel[0]
        dy = reference_location[1] - brightest_pixel[1]

        # Créer une matrice de translation et aligner l'image
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_img = cv2.warpAffine(img, translation_matrix, (w, h))

        aligned_images.append(aligned_img)

    return aligned_images

def apply_mean_mask_subtraction(image, k_size=3):
    """
    Applique une soustraction de masque moyen à l'image avec des noyaux croissants.

    :param image: Image d'entrée (autocorrelogramme).
    :param max_kernel_size: Taille maximale du noyau carré (doit être impair).
    :return: Image traitée.
    """
    processed_image = image.copy()
    # Créer un noyau moyen
    kernel = np.ones((k_size, k_size), np.float32) / (k_size * k_size)

    # Appliquer le filtre moyenneur
    mean_filtered = cv2.filter2D(image, -1, kernel)

    # Soustraire le résultat filtré de l'image originale
    processed_image = cv2.subtract(processed_image, mean_filtered)

    return processed_image


st.title("Data exploration for binaries stars post processing")

image_files = []
dir = "images"
for file in listdir(dir):
    file_path = dir + '/' + file
    if isfile(file_path) and splitext(file_path)[1]=='.fit':
        image_files.append(file_path)

print(len(image_files))
    
# Liste des chemins de fichiers des images de tavelures
#image_files = ['images/speckle1.jpg', 'images/speckle2.jpg', 'images/speckle3.jpg'] # Remplacez par vos fichiers

# Traitement des images de speckle
process_speckle_interferometry(image_files)
