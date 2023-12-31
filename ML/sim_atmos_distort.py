# Includes
import matplotlib.pyplot as plt
import numpy as np
from classes_atmos_sim import atmospheric_simulation
import random
from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from processing import AstroImageProcessing

# Size of simulated image
nxy = 256
number_simulated_pictures=50
stars_simulated=1200
result=[]
for stars in range(stars_simulated):
      # Binary star specs
  rho = random.uniform(0.35,0.9) # Set separation in arcseconds
  phi = int(random.uniform(0,360)) # Set angle in degrees  
  print(rho,phi)
  for j in range(number_simulated_pictures):
    # Instantiate simulation object
    sim = atmospheric_simulation()

    ## Assign all variables of simulation
    # Aperture/Telescope Specifications:
    sim.diameter_m = 2.133 # Mirror diameter in meters
    sim.focal_length = 129.69 # Effective focal length in meters

    # Camera Specs
    sim.wavelength = 0.8E-6 # Wavelength of light
    sim.pixel = 8E-6 # length of pixel side in m
    sim.bits = 14 # Bits in camera ADC
    sim.nxy = nxy # Length of sensor side in pixels
    sim.center = int(nxy/2) # Center of sensor in pixels
    sim.platescale = 206265*sim.pixel/(sim.focal_length) # Calculate plate scale
    sim.gamma = 1.6 # Gamma correction

    # Binary star specs
    sim.rho = rho # Set separation in arcseconds
    sim.phi = phi # Set angle in degrees  

    # Atmospheric Specs
    sim.alpha = 1/(100+random.uniform(-10, 10)) # Multiplicative constant
    sim.r0 = 0.2 + random.uniform(-0.05, 0.05)# Fried Parameter

    # Initializing Empty Images
    sim.input_img = np.zeros((nxy,nxy)) # Input binary star object
    sim.aperture_screen_s = np.zeros((nxy,nxy)) #  Aperture screen
    sim.phase_screen = np.zeros((nxy,nxy)) #  Phase screen values (unwrapped)
    sim.atmosphere_screen = np.zeros((nxy,nxy)) #  Atmosphere screen
    sim.pupil_screen = np.zeros((nxy,nxy)) # Total Pupil Screen
    sim.psf = np.zeros((nxy,nxy)) # PSF of aperture/atmosphere
    sim.binary_img = np.zeros((nxy,nxy)) # Simulated Binary Image

    ## Run Simulation
    # Create ideal binary star image
    sim.create_input_image()
    # Create telescope aperture image
    sim.create_aperture()
    # Create atmospheric phase screen image
    sim.create_atmosphere_screen()
    # Calculate PSF of aperture and atmospheric phase screen
    sim.get_psf()
    # Calculate simulated binary star image from PSF
    sim.get_binary()


    ## Adding photon shot noise
    # Photon numbers to test
    photons = (1000,100000)
    # Number of tests to run
    photons_n = len(photons)
    # Shot Noise Images
    img_shot_noise = np.zeros((photons_n,nxy,nxy))
    # Loop through photon values
    for i in np.arange(photons_n):
      # Add Poisson Noise
      img_shot_noise[i] = sim.add_noise(sim.binary_img,photons=photons[i],gaussian_var=0)

    ## Comparing effects of shot, gaussian, and gaussian/shot noise
    # Gaussian noise variance (in photons)
    var = 3
    # Shot noise number of received photons
    photons2 = 50E3
    # Noise Images
    img_shot_noise2 = np.zeros((nxy,nxy))
    img_gaussian_noise = np.zeros((nxy,nxy)) 
    img_gaussian_shot_noise = np.zeros((nxy,nxy))  
    # Calculate Noise Images
    binary_img_scaled = photons2*(np.abs(sim.binary_img)/np.sum(sim.binary_img))
    img_shot_noise2 = sim.add_noise(binary_img_scaled,photons=photons2,gaussian_var=0)
    img_gaussian_shot_noise = sim.add_noise(binary_img_scaled,photons=photons2,gaussian_var=var)

      
    ##Plots
    colormap = "jet"


    result.append(sim.binary_img.copy())

  psd = []
  for im in result:
    psd.append(np.square(np.abs(AstroImageProcessing.fourier_transform(im))))

  psd_average= AstroImageProcessing.average_images((psd))
  spatial = AstroImageProcessing.inverse_fourier_transform((psd_average))
  spatial = spatial.real
  img = np.mean(result, axis=0)
  image_combinee = np.stack((spatial, img), axis=-1)
  print(image_combinee.shape)

  #img = np.mean(result, axis=0)
  #np.save(f"simulated/y_{stars+350}.npy",sim.input_img)
  np.save(f"simulated/ex_{stars+888}.npy",image_combinee)
  np.save(f"simulated/ec_{stars+888}.npy",np.array([rho, phi*np.pi/180]))
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

