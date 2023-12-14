# speckles-interferometry
Test on speckles interferometry for binaries stars resolution

# Principe
main.py calculate autocorrelation mean from fit files in directory

postprocess.py is a streamlite program that will load the results from main.py and calculate a mean filter for removing noise, then try to find an ellipse that fit the figure, use the angle of this ellipse to slice the image to obtain a 1D curve on which I use gradient to isolate peaks and then calculate the distance between peaks.

![Alt text](https://github.com/air01a/speckles-interferometry/blob/main/doc/streamlit.png?raw=true "sun") 

Other scripts are tests and ML test models (WIP)