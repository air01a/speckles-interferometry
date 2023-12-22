import math
import numpy as np
import cv2

def calculate_line(point_a, point_b, variance=0.0):
     
    (x1,y1) = point_a
    (x2,y2) = point_b


    m = (y2-y1)/(x2-x1)*(1+variance*5)
    p = y1-m*x1
    return(m,p)


def calculate_perp(m, point):
    (x,y) = point
    m_perp = -1 / m
    p = y -m_perp * x
    return (m_perp,p)

def detect_and_remove_lines(image):

    (h,w) = image.shape
    for i in range(int(w/2-0.3*w),int(w/2+0.3*w)):
        image[int(h/2), i] = (image[int(h/2)-1, i] + image[int(h/2)+1, i])/2
        image[i, int(h/2)] = (image[i,int(h/2)-1] + image[i,int(h/2)+1])/2
    return image



def circle_calculation(pts):
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


def angle_with_reference(centre, reference, point):
    vector1 = reference - centre
    vector2 = point - centre
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def erase_principal_disk(image, vector):

    # Exemple de profil d'intensité pour le disque central
    # Supposons que c'est un profil d'intensité linéaire pour simplifier l'exemple
    # Taille de l'image
    (image_size_x,image_size_y) = image.shape

    # Créer une matrice pour le disque central
    central_disk = np.zeros((image_size_x, image_size_y))

    # Calculer la distance de chaque point au centre de l'image
    centre_x = image_size_x // 2
    centre_y = image_size_y // 2
    for x in range(image_size_x):
        for y in range(image_size_y):
            distance = np.sqrt((x - centre_x)**2 + (y - centre_y)**2)
            distance_index = int(distance)

            # Utiliser le profil d'intensité pour définir la valeur du disque
            if distance_index < len(vector):
                central_disk[x, y] = vector[distance_index]
    #st.write(disque_central)
    #show_image(disque_central,"central disk",st)
    image = image - central_disk
    image[image<0]=0
    return (image, central_disk)

def find_contours(src):
    image=(src.copy()/src.max() * 255).astype(np.uint8)
    image = cv2.bilateralFilter(image, 15, 75, 75) 
    ret, thresh = cv2.threshold(image, 0.8, 255, 0)
    contours, hierarchy = cv2.findContours((thresh).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours=sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def find_ellipse(contour):
    ellipse = cv2.fitEllipse(contour)
    centre, axes, angle = ellipse
    axe1, axe2 = axes[0] / 2, axes[1] / 2
    major_axe = max(axe1,axe2)
    minor_axe = min(axe1,axe2)
    c = np.sqrt(major_axe**2 - minor_axe**2)
    angle_rad = np.deg2rad(angle-90)
    focus1 = (int(centre[0] + c * np.cos(angle_rad)), int(centre[1] + c * np.sin(angle_rad)))
    focus2 = (int(centre[0] - c * np.cos(angle_rad)), int(centre[1] - c * np.sin(angle_rad)))


    return (ellipse, major_axe, minor_axe, angle_rad, focus1, focus2)


def get_vector_from_direction(image, m, point, max_range):
    
    x_C, y_C = point
    dx = 1 / math.sqrt(1 + m**2)  # Normalized
    dy = m * dx
    step=1
    vec = []
    for d in range(max_range):        
        x_C2_1 = int(x_C + d/step * dx)
        y_C2_1 = int(y_C + d/step * dy)
        if x_C2_1<image.shape[0] and y_C2_1<image.shape[1]:
            vec.append(image[y_C2_1,x_C2_1])

    return vec


def slice_from_direction(image, m, point):
    image = image.copy()

    d_max = math.sqrt(image.shape[0]**2 + image.shape[1]**2) / 2

    dx = 1 / math.sqrt(1 + m**2)  # Normalized
    dy = m * dx

    curve = []
    step = 1

    x_C,y_C = point
    for d in range(0, int(d_max)*step):
        # Calculate 2 points
        # with parametric formulation of a line
        x_C2_1 = x_C + d/step * dx
        y_C2_1 = y_C + d/step * dy

        x_C2_2 = x_C - d/step * dx
        y_C2_2 = y_C - d/step * dy

        if 0 <= x_C2_1 < image.shape[0] and 0 <= y_C2_1 < image.shape[1]:
            curve.append([d/step,image[int(y_C2_1),int(x_C2_1)]])

        
        if 0 <= x_C2_2 < image.shape[0] and 0 <= y_C2_2 < image.shape[1]:
            curve.append([-d/step,image[int(y_C2_2),int(x_C2_2)]])
    

    curve = sorted(curve, key=lambda pair: pair[0])
    return curve

def find_peaks(curve):
    
    #y = y-smoothed_y_values


    grads = []
    x2=[]
    peaks=[]
    in_peak=False
    first_peak=-1
    for i in range(1,len(curve)-1):
        x2.append(i)
        grad = 0.0 + (curve[i+1][1]-curve[i][1])
        if grad>0:
            in_peak=True
            first_peak=-1
            
        if grad<0:
            if in_peak:
                if first_peak!=-1:

                    ind = i-(i - first_peak)//2
                else:
                    ind=i

                    
                peaks.append([ind,curve[ind][1]])
            first_peak = -1
            in_peak=False


        if grad==0 and in_peak and first_peak==-1:
            first_peak = i

        last = grad
        grads.append(grad)

    peaks = sorted(peaks, key=lambda pair: pair[1], reverse=True)
    return peaks,grads