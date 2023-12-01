import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import splitext, isfile, isdir
from imagemanager import ImageManager
from processing import AstroImageProcessing
from pyplot_utils import show_image, show_image_3d

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

def load_speckle_images(image_files):
    imager = ImageManager()
    tab =  [roi for roi in [AstroImageProcessing.find_roi(imager.read_image(file)) for file in image_files] if roi is not None]
    max_w = 0
    max_h = 0
    for im in tab:
        (h, w) = im.shape
        max_w = max(w, max_w)
        max_h = max(h, max_h)

    output = []
    for index, im in enumerate(tab):
        if tab[index] is not None and len(tab[index]>0):
            output.append(AstroImageProcessing.resize_with_padding(im, max_w, max_h))
    return output


def process_speckle_interferometry(image_files,name):
    """ Traite un ensemble d'images de tavelures pour la speckle interferometry. """
    # Charger les images

    speckle_images = load_speckle_images(image_files)
    print(f"    * images loaded : {len(speckle_images)}")
    fouriers = []
    psd = []
    for im in speckle_images:
        fouriers.append(AstroImageProcessing.fourier_transform(im))
        psd.append(np.square(np.abs(AstroImageProcessing.fourier_transform(im))))
    
    psd_average= AstroImageProcessing.average_images((psd))
    spatial = AstroImageProcessing.inverse_fourier_transform((psd_average))
    spatial = spatial.real

    #show_image(spatial,"Observed and expected secondary locations")
    np.save(f"results/{name}.npy",spatial)
    print(f"    * Writing results/{name}.npy")
    spatial = (spatial)/(spatial.max())*65535
    cv2.imwrite(f"results/{name}.png",spatial.astype(np.uint16))
    #show_image_3d(spatial)

    
    


maindir = "./imagesrepo/"

for dr in listdir(maindir):
    if isdir(maindir+dr):

        image_files = []
        dir = maindir+dr+"/"
        for file in listdir(dir):
            file_path = dir + '/' + file
            if isfile(file_path) and splitext(file_path)[1]=='.fit':
                image_files.append(file_path)
        print(f"############\nProcessing {dir}")
        process_speckle_interferometry(image_files,dr)





"""
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

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(image)
        return maxLoc

    def align_images(images, reference_location):

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

    def filter(image):
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        radius = 30  # Radius of the low-pass filter
        mask = np.zeros((rows, cols), np.uint8)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
        mask[mask_area] = 1


        f_shift_filtered = image * mask
        return f_shift_filtered
"""