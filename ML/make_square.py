# Includes
import matplotlib.pyplot as plt
import numpy as np
from classes_atmos_sim import atmospheric_simulation
import random
from pathlib import Path
import sys
import math
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from processing import AstroImageProcessing
import cv2

# Size of simulated image
nxy = 256
number_simulated_pictures=50
stars_simulated=1200
result=[]
for stars in range(stars_simulated):
      # Binary star specs
  x1 = int(random.uniform(10.0,nxy-10)) # Set separation in arcseconds
  y1 = int(random.uniform(10.0,nxy-10)) ## Set angle in degrees  
  x2 = int(random.uniform(10.0,nxy-10)) # Set separation in arcseconds
  y2 = int(random.uniform(10.0,nxy-10)) ## Set angle in degrees  
  
  dist=math.sqrt((x2-x1)**2 + (y2-y1)**2)
  print(x1,y1,x2,y2,dist)
  im = np.zeros((nxy,nxy))
  color = 255  # Bleu
  thickness = -1 

  im = cv2.rectangle(im, (x1-10,y1-10), (x1+10,y1+10), color, thickness)
  im = cv2.rectangle(im, (x2-10,y2-10), (x2+10,y2+10), color, thickness)
  #plt.imshow(im)
  #plt.show()

  #img = np.mean(result, axis=0)
  #np.save(f"simulated/y_{stars+350}.npy",sim.input_img)
  np.save(f"simulated2/x_{stars}.npy",im)
  np.save(f"simulated2/c_{stars}.npy",np.array([x1,y1,x2,y2]))
  #print(sim.x1,sim.y1,sim.x2,sim.y2)
  #plt.imshow(image_combinee)
  #plt.show()
  

"""
plt.imshow(sim.atmosphere_screen.real)
plt.show()
plt.figure(figsize = (18,8))
plt.subplot(1,2,1)
plt.imshow(sim.phase_screen, cmap=colormap)
plt.colorbar()
plt.title("Complex Pupil Phase")
plt.subplot(1,2,2)
plt.imshow(sim.aperture_screen_s, cmap=colormap)
plt.title("Complex Pupil Magnitude")

plt.figure(figsize = (8,8))
plt.imshow(sim.psf, cmap=colormap)
plt.title("Total System PSF")

plt.figure(figsize = (14,6))
plt.subplot(1,2,1)
plt.imshow(sim.emphasized_image(sim.input_img), cmap=colormap, extent = (-6.513,6.513,-6.513,6.513))
plt.xlabel("[arcsec]")
plt.ylabel("[arcsec]")
plt.title("Input Image")
plt.subplot(1,2,2)
plt.imshow(sim.binary_img, cmap=colormap)
plt.xlabel("[pixels]")
plt.ylabel("[pixels]")
plt.title("Binary Image")

plt.figure(figsize = (10+10*photons_n,10), dpi = 50)
plt.subplot(1,1+photons_n,1)
plt.imshow(sim.binary_img, cmap=colormap)
plt.title("Sensor Image (No Shot Noise)")
for i in np.arange(2):
  plt.subplot(1,1+photons_n,2+i)
  plt.imshow(img_shot_noise[i])
  plt.title(("Sensor Image (" + str(photons[i]) + " Photons)"))

plt.figure(figsize = (14,6))
plt.subplot(1,2,1)
plt.title("Shot Noise")
plt.imshow(img_shot_noise2)
plt.subplot(1,2,2)
plt.title("Shot + Gaussian Read Noise")
plt.imshow(img_gaussian_shot_noise)
  
plt.show()

"""

