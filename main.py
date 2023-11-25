import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import splitext, isfile, isdir, getsize
from imagemanager import ImageManager
from processing import AstroImageProcessing


def load_speckle_images(image_files):
    """ Charge les images de tavelures à partir d'une liste de chemins de fichiers. """
    imager = ImageManager()

    tab =  [(AstroImageProcessing.levels(AstroImageProcessing.stretch(imager.read_image(file),0.2),5000,0.1,65000)) for file in image_files]
    i=0
    for t in tab:
        cv2.imshow(f"test{i}",(t/255).astype(np.uint8))
        i+=1
    return tab

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
    return np.mean(speckle_images, axis=0)

def fourier_transform(image):
    """ Applique la transformation de Fourier à une image. """
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    return 20*np.log(np.abs(f_shift))

def process_speckle_interferometry(image_files):
    """ Traite un ensemble d'images de tavelures pour la speckle interferometry. """
    # Charger les images
    speckle_images = load_speckle_images(image_files)

    # Aligner les images
    aligned_images = align_images_using_cross_correlation(speckle_images[0], speckle_images[1:])

    # Calculer l'image moyenne
    average_image = average_speckle_images(aligned_images)

    # Appliquer la transformation de Fourier
    fourier_image = fourier_transform(average_image)

    # Afficher les résultats
    plt.subplot(121),plt.imshow(average_image, cmap = 'gray')
    plt.title('Image Moyenne'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(fourier_image, cmap = 'gray')
    plt.title('Transformée de Fourier'), plt.xticks([]), plt.yticks([])

    plt.show()

image_files = []
dir = "images"
for file in listdir(dir):
    file_path = dir + '/' + file
    if isfile(file_path) and splitext(file_path)[1]=='.fit':
        image_files.append(file_path)
    
# Liste des chemins de fichiers des images de tavelures
#image_files = ['images/speckle1.jpg', 'images/speckle2.jpg', 'images/speckle3.jpg'] # Remplacez par vos fichiers

# Traitement des images de speckle
process_speckle_interferometry(image_files)