import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from processing import AstroImageProcessing
from pyplot_utils import show_image, show_image_3d
import cv2

from skimage.feature import peak_local_max

import numpy as np
from scipy.optimize import minimize
import math

def calculate_circle_center(points):
    # Fonction pour calculer le carré de la distance d'un point au centre du cercle
    def dist_squared(center, point):
        return (center[0] - point[0])**2 + (center[1] - point[1])**2

    # Fonction objectif : minimiser la variance des distances des points au centre estimé
    def objective(center, points):
        distances = [dist_squared(center, p) for p in points]
        return np.var(distances)

    # Calcul des moyennes comme point de départ pour l'optimisation
    x_mean = np.mean([p[0] for p in points])
    y_mean = np.mean([p[1] for p in points])
    initial_guess = [x_mean, y_mean]

    # Optimisation pour trouver le centre du cercle
    result = minimize(objective, initial_guess, args=(points,))
    return result.x


image = cv2.imread("results/stf2380_output.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

show_image_3d(image)
# Création d'une image aléatoire de taille 96x96
image_size = image.shape[0]
#image = np.random.randint(0, 256, (image_size, image_size))

# Calcul de la valeur moyenne des pixels
mean_value = np.mean(image)

# Application de l'algorithme K-means avec 2 clusters

imagec = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

ret, thresh = cv2.threshold(image, image.mean()*0.5, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ellipse = cv2.fitEllipse(contours[0])
centre, axes, angle = ellipse
axe1, axe2 = axes[0] / 2, axes[1] / 2
axe_majeur = max(axe1,axe2)
axe_mineur = min(axe1,axe2)
c = np.sqrt(axe_majeur**2 - axe_mineur**2)
angle_rad = np.deg2rad(angle-90)

# Calculer les coordonnées des foyers
foyer1 = (int(centre[0] + c * np.cos(angle_rad)), int(centre[1] + c * np.sin(angle_rad)))
foyer2 = (int(centre[0] - c * np.cos(angle_rad)), int(centre[1] - c * np.sin(angle_rad)))
print("Centre:", centre)
print("Foyer 1:", foyer1)
print("Foyer 2:", foyer2)
print(f"Angle {angle-90}")

"""cv2.ellipse(imagec,ellipse,(0,255,0),2)
cv2.circle(imagec, (foyer1[0],foyer1[1]), 5, (30,255,30), -1)
cv2.circle(imagec, (foyer2[0],foyer2[1]), 5, (30,255,30), -1)
cv2.circle(imagec, (int(centre[0]),int(centre[1])), 5, (255,30,30), -1)
#imagec = cv2.drawContours(imagec, contours, -1, (0,255,0), 3)
show_image(imagec)




kmeans = KMeans(n_clusters=3, random_state=0,n_init=100).fit(data)

# Affichage des centres des clusters
print(kmeans.cluster_centers_)
image = ((image-image.min())/(image.max()-image.min()) * 255).astype(np.uint8)
imagec = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)"""

"""
ret, thresh = cv2.threshold(image, image.mean()*1.5, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagec, contours, -1, (0,255,0), 3)
"""

imagec_save = imagec.copy()

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(imagec, (x, y), (x + w, y + h), (0, 255, 0), 2)
    x_bnd = x
    y_bnd = y


"""
for center in kmeans.cluster_centers_:
    (cx,cy)=(int(center[0]),int(center[1]))
    imagec = cv2.circle(imagec, (cx,cy), 5, (255,30,30), 1)

show_image(imagec)
imagec=imagec_save.copy()

data = []
local_center=[]
coordinates = peak_local_max(image, min_distance=1)

main_vecx = abs(foyer1[0]-centre[0])
main_vecy = abs(foyer1[1]-centre[1])

#cv2.line(imagec,(int(centre[0]),int(centre[1])),(int(main_vecx+centre[0]), int(main_vecy+centre[1])),(255,0,0),5)

centrums = []
for coord in coordinates:
    if image[coord[0],coord[1]]>image.mean():
        x = coord[1]
        y = coord[0]
        dist = math.sqrt((x-centre[0])**2+(y-centre[1])**2)   
        if  dist < axe_majeur/2:
        
            #imagec = cv2.circle(imagec, (x,y), 5, (255,30,30), 1)

            imagec[coord[0],coord[1]] = [255,255,0]
            data.append([coord[0],coord[1]])

            print("dist:%f, value: %i" % (dist,image[coord[0],coord[1]]))

            x1 = abs(coord[0] - centre[0])
            y1 = abs(coord[1] - centre[1])
            scalaire = abs(main_vecx*x1 + main_vecy*y1)
            print(f"scalire : {scalaire}")
            centrums.append([scalaire, dist, coord[0], coord[1]]) 

            ### TODO
            ### CALCULATE MAIN AXIS EQUATION, FIND BEST CENTER ACCORDING TO THIS LINE
centrums = sorted(centrums, key=lambda x: x[0], reverse=True)
print(centrums)
cv2.circle(imagec, (centrums[0][2],centrums[0][3]), 1, (255,0,30), 1)

kmeans = KMeans(n_clusters=3, random_state=0,n_init=100).fit(data)
for center in kmeans.cluster_centers_:
    (cx,cy)=(int(center[0]),int(center[1]))
    imagec = cv2.circle(imagec, (cx,cy), 5, (255,255,30), 1)
cv2.ellipse(imagec,ellipse,(0,255,0),2)"""

x1 = foyer1[0]
y1 = foyer1[1]

m = (y1-centre[1])/(x1-centre[0])
p = y1-m*x1

for x0 in range(x_bnd, x_bnd+w):
    y0 = int(m*x0+p)
    cv2.circle(imagec, (x0,y0), 1, (255,0,30), 1)
    print(image[y0,x0])

#print(coordinates)
show_image(imagec)



#imagec = imagec_save.copy()
imagec = cv2.circle(imagec, (int(centre[0]),int(centre[1])), int(axe_mineur/3), (0,0,0), -1)

gray=cv2.cvtColor(imagec, cv2.COLOR_BGR2GRAY)

data = []
for y in range(gray.shape[0]):
    for x in range(gray.shape[0]):
        if gray[y, x] > gray.mean():
            data.append([x, y])
            print(x,y)

data = np.array(data)
kmeans = KMeans(n_clusters=2, random_state=0,n_init=10).fit(data)
for center in kmeans.cluster_centers_:
    (cx,cy)=(int(center[0]),int(center[1]))
    imagec = cv2.circle(imagec, (cx,cy), 5, (255,30,30), 1)

show_image(imagec)
