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
from scipy.signal import find_peaks

from imagecalculation import  find_ellipse
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from math import sqrt

from findpeaks import findpeaks

def detect_peaks2(image):

    fp = findpeaks(method='mask',denoise="mean")

    res=(fp.fit(image)['Xdetect']*image)
    (x,y) = find_brightest_pixel(res)
    res[y,x]=0
    (x2,y2) = find_brightest_pixel(res)
    print(x,y)
    print(x2,y2)
    print(f"rho { sqrt((x-x2)**2+(y-y2)**2)}")
    #fp.plot_preprocessing()
    #fp.plot()
    cv2.circle(image,(x2,y2),2,image.max()*1.1,-1)
    cv2.circle(image,(x,y),2,image.max()*1.1,-1)
    #-show_image(image)
    return (x,y, x2,y2)
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background


    rest = detected_peaks*image
    output = np.zeros_like(image)
    peak_number=40
    for i in range(peak_number):
        max_loc = find_brightest_pixel(rest)
        rest[max_loc[1],max_loc[0]]=0
        output[max_loc[1],max_loc[0]]=peak_number-i
    """


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background


    rest = detected_peaks*image
    output = np.zeros_like(image)
    peak_number=40
    for i in range(peak_number):
        max_loc = find_brightest_pixel(rest)
        rest[max_loc[1],max_loc[0]]=0
        output[max_loc[1],max_loc[0]]=peak_number-i



    #show_image(output)


    return output

def load_speckle_images(image_files):
    imager = ImageManager()
    tab =  [roi for roi in [AstroImageProcessing.find_roi((imager.read_image(file))) for file in image_files] if roi is not None]
    max_w = 0
    max_h = 0
    for im in tab:
        (h, w) = im.shape
        max_w = max(w, max_w)
        max_h = max(h, max_h)

    output = []
    peak = np.zeros((max_h,max_w))
    for index, im in enumerate(tab):
        if tab[index] is not None and len(tab[index]>0):
            output.append(AstroImageProcessing.resize_with_padding(im, max_w, max_h))
            peak += detect_peaks(output[-1])
            
            
    #-show_image(peak)
    
    return output

def find_brightest_pixel(image):
    """ Trouve la position du pixel le plus lumineux dans l'image. """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
    return max_loc

def align_images(images):
    """ Aligner les images en s'assurant qu'elles ont toutes la même taille finale. """
    ref = images[0]
    
    #brightest_pixels = [find_brightest_pixel(img) for img in images]

    brightest_pixels  = [align_images_2(ref,img) for img in images]
    # Trouver les décalages maximaux pour aligner les images
    max_x = max([p[0] for p in brightest_pixels])
    max_y = max([p[1] for p in brightest_pixels])

    # Calculer les dimensions finales nécessaires
    max_width = max([img.shape[1] + (max_x - x) for img, (x, y) in zip(images, brightest_pixels)])
    max_height = max([img.shape[0] + (max_y - y) for img, (x, y) in zip(images, brightest_pixels)])

    aligned_images = [ref]
    for img, (x, y) in zip(images, brightest_pixels):
        dx, dy = max_x - x, max_y - y

        # Créer une nouvelle image avec les dimensions finales
        new_image = np.zeros((max_height, max_width), dtype=img.dtype)
        new_image[dy:dy+img.shape[0], dx:dx+img.shape[1]] = img
        aligned_images.append(new_image)

    return aligned_images


def align_images_2(image1, image2):
    # Convertir les images en niveaux de gris
    gray1 = ((image1-image1.min())/(image1.max() - image1.min())*255).astype(np.uint8)
    gray2 = ((image2-image2.min())/(image2.max() - image2.min())*255).astype(np.uint8)

    # Calculer l'intercorrélation entre les deux images
    correlation_output = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    
    # Trouver la position du maximum de corrélation
    _, _, _, max_loc = cv2.minMaxLoc(correlation_output)

    # Calculer le décalage entre les images
    #x_offset = max_loc[0]
    #y_offset = max_loc[1]

    # Aligner la deuxième image sur la première
    #aligned_image = cv2.warpAffine(image2, np.float32([[1, 0, x_offset], [0, 1, y_offset]]), (image1.shape[1], image1.shape[0]))

    return max_loc

def sum_aligned_images(images):
    """ Somme des images alignées. """
    aligned_images = align_images(images)
    sum_image = np.zeros_like(aligned_images[0]).astype(np.float32)
    i=0
    for img in aligned_images:
        # Évite la saturation
        sum_image=sum_image+img
        i+=1

    return sum_image


def find_contours(src,mean_filter=9):
    image = ((src.copy() - src.min())/(src.max()-src.min())*255).astype(np.uint8)
    ret, thresh = cv2.threshold(image,mean_filter*image.mean(), 255, 0)
    #show_image(thresh, "test")
    #show_image(thresh)
    contours, hierarchy = cv2.findContours((thresh).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours=sorted(contours, key=cv2.contourArea, reverse=True)
    return contours

def get_angle(image):

    contours = find_contours(image)
    if len(contours[0])>5:
        (ellipse, major_axe, minor_axe, angle_rad, focus1, focus2) = find_ellipse(contours[0])
    else:
        return None, None
    image=cv2.ellipse(image.copy(), ellipse,255,1)
    #show_image(image, "test")
    return (angle_rad,major_axe)

def isolate_peaks2(image):
    # Appliquer un flou pour réduire le bruit (optionnel)
    gray = (255*(image - image.min())/(image.max()-image.min())).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #-show_image(blurred)
    #-show_image_3d(blurred)
    return detect_peaks2(blurred)


def process_speckle_interferometry(image_files,name):
    """ Traite un ensemble d'images de tavelures pour la speckle interferometry. """
    # Charger les images
    angles =  []
    axes = []
    angles_index = []
    speckle_images = load_speckle_images(image_files)
    
    for index, im in enumerate(speckle_images):
        angles_index.append([im.max(), im])
    sorted_result = sorted(angles_index, key=lambda x: x[0], reverse=True)
    #print(sorted_result)


    best_images = []
    for i in range(0,300):
        #best_images.append(speckle_images[sorted_result[i][1]])
        best_images.append(sorted_result[i][1])

    speckle_images = best_images
    #sum_aligned_images(speckle_images),3)
    speckle_images = (sum_aligned_images(speckle_images))
    brightest_pixel = find_brightest_pixel(speckle_images)

    #speckle_images[brightest_pixel[1],brightest_pixel[0]]=0
    #speckle_images[brightest_pixel[1],brightest_pixel[0]]=speckle_images.max()

    speckle_images = AstroImageProcessing.apply_mean_mask_subtraction(speckle_images,7)


    speckle_images[speckle_images<0]=0


    #for im in align2_images(speckle_images):
    im = speckle_images
    #for i in range(0,10):
    np.save('test.npy',im)
    im = np.clip(im,im.mean()*5/2, im.max())
    show_image(im,"test "+str(i))
    #show_image_3d(im)
    
    x,y,x2,y2 = isolate_peaks2(speckle_images)
    return x,y,x2,y2



results = []
maindir = "imagesrepo/"

for dr in listdir(maindir):
    if isdir(maindir+dr):

        image_files = []
        dir = maindir+dr+"/"
        for file in listdir(dir):
            file_path = dir + '/' + file
            if isfile(file_path) and splitext(file_path)[1]=='.fit':
                image_files.append(file_path)
        print(f"############\nProcessing {dir}")
        x,y, x2,y2 = process_speckle_interferometry(image_files,dr)
        results.append([dir,x,y,x2,y2])
        

for result in results:
    name,x,y,x2,y2 = result
    rho = ((x-x2)**2 + (y-y2)**2)**(0.5)*0.099
    print(f"{name}, {rho}")





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