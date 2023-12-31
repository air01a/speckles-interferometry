# Includes
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.signal import fftconvolve
from scipy.signal import argrelextrema
import cv2

## Parameters
# Image specs
nxy = 256
center = int(nxy/2)

class atmospheric_simulation():
  def __init__(self):
    # Aperture/Telescope Specifications:
    self.diameter_m = None # Mirror diameter in meters
    self.focal_length = None # Effective focal length in meters

    # Camera Specs
    self.wavelength = None # Wavelength of light
    self.pixel = None # length of pixel side in m
    self.bits = None # Bits in camera ADC
    self.nxy = None # Length of sensor side in pixels
    self.center = None # Center of sensor in pixels
    self.platescale = None # Plate scale
    self.gamma = None # Gamma correction

    # Binary star specs
    self.rho = None # Set separation in arcseconds
    self.phi = None # Set angle in degrees  
    
    # Atmospheric Specs
    self.alpha = None # Multiplicative constant
    self.r0 = None # Fried Parameter
    # NOTE: R0 DOESN'T YET PROPERLY MODEL A REAL R0 VALUE, ALPHA MUST BE 
    # PROPERLY SET TO GIVE CORRECT COMPARISON TO REAL R0 VALUES. DID NOT 
    # HAVE TIME TO CALCULATE THIS VALUE
    
    # Initializing Empty Images
    self.input_img = None # Input binary star object
    self.aperture_screen_s = None #  Aperture screen
    self.phase_screen = None #  Phase screen values (unwrapped)
    self.atmosphere_screen = None #  Atmosphere screen
    self.pupil_screen = None # Total Pupil Screen
    self.psf = None # PSF of aperture/atmosphere
    self.binary_img = None # Simulated Binary Image
    self.x1 = 0
    self.y1 = 0
    self.x2 = 0
    self.y2 = 0
    
  def create_input_image(self): 
    # Calculate coordinates of stars
    x = int( self.rho/(2*self.platescale) * np.cos(np.deg2rad(self.phi)) )
    y = int( self.rho/(2*self.platescale) * np.sin(np.deg2rad(self.phi)) )
    x1 = self.center + x
    y1 = self.center + y
    x2 = self.center - x
    y2 = self.center - y    
    (self.x1, self.y1) = (x1, y1)
    (self.x2, self.y2) = (x2,y2)
    cv2.circle(self.input_img, (x1,y1),5,150,-1)
    cv2.circle(self.input_img, (x2,y2),5,150,-1)
    # Place stars on image
    self.input_img[y1,x1] = 1 
    self.input_img[y2,x2] = 1
    # Scale image power to 1
    input_img_power = np.sum(np.power(self.input_img,2))
    self.input_img = np.divide(self.input_img,np.sqrt(input_img_power))

  def emphasized_image(self, img, circle_radius = 3):
    # Convolve small circles with input images to make them more visible
    # Create meshgrid
    xx,yy = np.meshgrid(np.arange(self.nxy),np.arange(self.nxy))
    # Calculate grid of distances from circle center
    radius = (xx-center) ** 2 + (yy-center) ** 2 
    # Draw boolean circle
    circle = (radius < (circle_radius**2)).astype(np.int64)
    # Convolve circle with difficult to see images
    img_emph = fftconvolve(img,circle)[self.center:self.center+self.nxy,self.center:self.center+self.nxy]
    # Return emphasized image
    return img_emph

    
  def create_aperture(self):
    ## Telescope aperture creation:
    # Total spatial sample range
    X_aperture_s = 1/self.pixel 
    # dx value for sampled aperture image
    dx_aperture_s = X_aperture_s/nxy 
    # Coordinates of sampled image
    x_aperture_s = np.arange(0,X_aperture_s,dx_aperture_s) - X_aperture_s/2
    # Meshgrid of sampled coordinates
    xx_s,yy_s = np.meshgrid(x_aperture_s,x_aperture_s)
    # Scaled aperture diameter to effectively resample aperture image
    diameter_s = self.diameter_m/(self.focal_length*self.wavelength)
    # Draw new circle at correct dimensions
    # Calculate grid of distances from circle center
    circle_s = (xx_s) ** 2 + (yy_s) ** 2 
    # Draw boolean circle
    circle_s = circle_s < (diameter_s/2)**2 
    # Convert boolean circle to int
    circle_s= circle_s.astype(np.int64)
    
    # Scale aperture image power to 1
    aperture_screen_power = np.sum(np.power(circle_s,2))
    # Save aperture image in units of meters
    self.aperture_screen_s = np.divide(circle_s,np.sqrt(aperture_screen_power))
    
    # Calculate effective size of sampled aperture image in meters
    self.X_aperture_s_meff = self.focal_length*self.wavelength/self.pixel
    
  def create_atmosphere_screen(self):
    ## Phase screen creation:
    # Generate random image
    # To be used in creating random atmospheric element
    phase_phase = np.multiply(np.pi,np.random.normal(loc=0,scale=1,size=(nxy,nxy)))
    # Total array sample size
    d_aperture = self.focal_length*self.wavelength/self.pixel
    # Spatial sample resolution
    dxy = d_aperture/nxy
    # Spatial frequency resolution
    df = 1/(d_aperture) 
    # Image sample indices array
    x = np.multiply( np.subtract(np.arange(self.nxy),self.center), dxy ) 
    # Spatial Frequency indices array
    xf = np.multiply( np.subtract(np.arange(self.nxy),self.center), df ) 
    # Meshgrid of spatial frequency domain
    [xx,yy]=np.meshgrid(xf,xf)
    # Radius from center meshgrid
    rr = (np.sqrt(np.power(xx,2)+np.power(yy,2)))
    # Calculate Kolmogorov spectral density
    phase_PSD = np.power(rr,-11/3)
    phase_PSD = np.multiply(self.alpha*0.023/(self.r0**(5/3)),phase_PSD)
    # Set DC component to 0 (previous calc attempts to set to 1/0)
    phase_PSD[self.center,self.center] = 0 
    # Construct phase screen spectrum
    phase_screen_f = np.multiply(np.sqrt(phase_PSD),np.exp(1j*phase_phase))
    # Calculate phase screen
    self.phase_screen = np.real(ifft2(fftshift(phase_screen_f)*nxy*nxy))
    # Create complex atmospheric screen
    self.atmosphere_screen = np.exp(np.multiply(1j,self.phase_screen))
    
  def get_psf(self):
    # Generate total screen, combining atmosphere and aperture
    self.pupil_screen = np.multiply(self.atmosphere_screen,self.aperture_screen_s)

    ## Calculate system's total response 
    # Calculate total PSF of system
    self.psf = fftshift(fft2(self.pupil_screen))
    self.psf = np.power(np.abs(self.psf),2)
    # Normalize PSF
    psf_power = np.sum(np.power(self.psf,2))
    self.psf = np.divide(self.psf,np.sqrt(psf_power))
    # Gamma correct PSF
    self.psf = np.power(self.psf,1/self.gamma) 
    # Normalize PSF
    psf_power = np.sum(np.power(self.psf,2))
    self.psf = np.divide(self.psf,np.sqrt(psf_power))
                      
  def get_binary(self):
    # Convolve PSF with input image using FFT
    self.binary_img = fftconvolve(self.input_img,self.psf)
    # Save the center 512x512 image
    self.binary_img = self.binary_img[self.center:self.center+self.nxy,self.center:self.center+self.nxy]
    
    # Normalize binary
    binary_power = np.sum(np.power(self.binary_img,2))
    self.binary_img = np.divide(self.binary_img,np.sqrt(binary_power))
    
  def add_noise(self,img,photons,gaussian_var):
    
    # Array to be returned
    img_noisy = np.array(img)
    
    # If photons > 0, add shot noise
    if (photons>0):
      # Normalize image to number of photons incident
      img_noisy = photons*(np.abs(img)/np.sum(img))
      # Calculate image with shot noise
      img_noisy = np.random.poisson(lam=img_noisy, size=None)
      
    # If gaussian_var > 0, add additive noise
    if (gaussian_var > 0):  
      # Create noise image
      noise = np.random.normal(loc=0,scale=gaussian_var,size=(self.nxy,self.nxy))
      # Add noise image to simulated image
      img_noisy = img_noisy+noise
      # Turn all negative pixels to 0
      img_noisy[img_noisy<0] = 0   
    
    # Return noisy image
    return img_noisy  
  
# Normalize image between 0 and a given max
def normalize(img, norm_max):
  img_new = np.array(img)
  img_new -= img.min()
  img_new = img_new*norm_max/img_new.max()
  return img_new