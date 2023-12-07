import types

from astropy.io import fits
from cv2 import mean as cv_mean
from cv2 import  imread, VideoCapture, CAP_PROP_FRAME_COUNT, cvtColor, COLOR_RGB2GRAY, \
    COLOR_BGR2RGB, COLOR_BayerGB2BGR, COLOR_BayerBG2BGR, THRESH_TOZERO, threshold, \
    GaussianBlur, Laplacian, CV_32F, COLOR_RGB2BGR, imwrite, convertScaleAbs, CAP_PROP_POS_FRAMES, \
    IMREAD_UNCHANGED, flip, COLOR_GRAY2RGB, COLOR_BayerRG2BGR, COLOR_BayerGR2BGR, \
    COLOR_BayerRG2BGR_VNG, COLOR_BayerGR2BGR_VNG, COLOR_BayerGB2BGR_VNG, COLOR_BayerBG2BGR_VNG, \
    COLOR_BayerRG2BGR_EA, COLOR_BayerGR2BGR_EA, COLOR_BayerGB2BGR_EA, COLOR_BayerBG2BGR_EA, resize, COLOR_BGR2GRAY
from numpy import max as np_max
from numpy import min as np_min
from numpy import sum as np_sum
from numpy import uint8, uint16, uint32, int32, float32, clip, zeros, float64, where, average, moveaxis, \
    unravel_index, ndarray, array,sort,frombuffer,reshape, append as np_append, empty, flip as np_flip

from os.path import splitext
from pathlib import Path
from os import path, remove, listdir, stat
import struct

class SerManager:
    def __init__(self,fname):
        self.fname=fname
        self.header=types.SimpleNamespace()
        with open(self.fname,"rb") as f:
            self.header.fileID=f.read(14).decode()
            self.header.luID=int.from_bytes(f.read(4), byteorder='little')
            self.header.colorID=int.from_bytes(f.read(4), byteorder='little')
            """
            Content:
                MONO= 0
                BAYER_RGGB= 8
                BAYER_GRBG= 9
                BAYER_GBRG= 10
                BAYER_BGGR= 11
                BAYER_CYYM= 16
                BAYER_YCMY= 17
                BAYER_YMCY= 18
                BAYER_MYYC= 19
                RGB= 100
                BGR= 101
            """
            if self.header.colorID <99:
                self.header.numPlanes = 1
            else:
                self.header.numPlanes = 3
                
            self.header.littleEndian=int.from_bytes(f.read(4), byteorder='little')
            self.header.imageWidth=int.from_bytes(f.read(4), byteorder='little')
            self.header.imageHeight=int.from_bytes(f.read(4), byteorder='little')
            self.header.PixelDepthPerPlane=int.from_bytes(f.read(4), byteorder='little')
            if self.header.PixelDepthPerPlane == 8:
                self.dtype = uint8
            elif self.header.PixelDepthPerPlane == 16:
                self.dtype = uint16
            self.header.frameCount=int.from_bytes(f.read(4), byteorder='little')
            self.header.observer=f.read(40).decode()
            self.header.instrument=f.read(40).decode()
            self.header.telescope=f.read(40).decode()
            self.header.dateTime=int.from_bytes(f.read(8), byteorder='little')
            self.imgSizeBytes = int(self.header.imageHeight*self.header.imageWidth*self.header.PixelDepthPerPlane*self.header.numPlanes/8)
            self.imgNum=0
        
    def get_img(self,imgNum=None):
        if imgNum is None:
            pass
        else:
            self.imgNum=imgNum
            
        with open(self.fname,"rb") as f:
            f.seek(int(178+self.imgNum*(self.imgSizeBytes)))
            frame = frombuffer(f.read(self.imgSizeBytes),dtype=self.dtype)
            
        self.imgNum+=1
        
        frame = reshape(frame,(self.header.imageHeight,self.header.imageWidth,self.header.numPlanes))
        return frame
    
    def int_to_little_indian(self, i):
        return struct.pack('<Q', i)
    
    def reduce_ser_file(self, output, frames_to_keep):
        with open(self.fname,"rb") as f:
            # Read  header
            header1 = f.read(38)
            f.read(4)  #Read framecount that will change according to len(frames_to_keep)
            header2 = f.read(136)

            with open(output,'wb') as f_out:
                f_out.write(header1)
                f_out.write(self.int_to_little_indian(len(frames_to_keep)))
                f_out.write(header2)

                for i in frames_to_keep:
                    f.seek(int(178+i*(self.imgSizeBytes)))
                    img = f.read(self.imgSizeBytes)
                    f_out.write(img)


    
class ImageManager:

    def to_gray(self, image):
        return cvtColor(image, COLOR_BGR2GRAY)


    def read_image(self,filename):
        """
        Read an image (in tiff, fits, png or jpg format) from a file.

        :param filename: Path name of the input image.
        :return: RGB or monochrome image.
        """

        name, suffix = splitext(filename)

        # Make sure files with extensions written in large print can be read as well.
        suffix = suffix.lower()
        # Case FITS format:
        if suffix in ('.fit', '.fits'):
            image = uint16(fits.getdata(filename))

            # FITS output file from AS3 is 16bit depth file, even though BITPIX
            # has been set to "-32", which would suggest "numpy.float32"
            # https://docs.astropy.org/en/stable/io/fits/usage/image.html
            # To process this data in PSS, do "round()" and convert numpy array to "np.uint16"
            if image.dtype == '>f4':
                image = image.round().astype(uint16)

            # If color image, move axis to be able to process the content
            if len(image.shape) == 3:
                ImageManager.debayer_frame(image)
                #image = moveaxis(image, 0, -1).copy()
            # Flip image horizontally to recover original orientation
            image = flip(image, 0)

        # Case other supported image formats:
        elif suffix in ('.tiff', '.tif', '.png', '.jpg'):
            input_image = imread(filename, IMREAD_UNCHANGED)
            if input_image is None:
                raise IOError("Cannot read image file. Possible cause: Path contains non-ascii characters")

            # If color image, convert to RGB mode.
     #       if len(input_image.shape) == 3:
     #           image = cvtColor(input_image, COLOR_BGR2RGB)
      #      else:
      # 
            image = input_image

        else:
            raise TypeError("Attempt to read image format other than 'tiff', 'tif',"
                            " '.png', '.jpg' or 'fit', 'fits'")
        return image
    



    def debayer_frame(frame_in, debayer_pattern='No change', debayer_method='Bilinear', BGR_input=False):
        '''
        Process a given input frame "frame_in", either containing one layer (B/W) or three layers
        (color) into an output frame "frame_out" as specified by the parameter "debayer_pattern".

        The rules for this transformation are:
        - If the "debayer_pattern" is "No change", the input frame is not changed, i.e. the
        output frame is identical to the input frame. The same applies if the input frame is of type
        "B/W" and the "debayer_pattern" is "Grayscale".
        - If the input frame is of type "color" and "debayer_pattern" is "Grayscale", the RGB / BGR
        image is converted into a B/W one.
        - If the input frame is of type "color", the "debayer_pattern" is "BGR" and "BGR_input" is
        'False', the "B" and "R" channels are exchanged. The same happens if "debayer_pattern" is "RGB"
        and "BGR_input" is 'True'. For the other two combinations the channels are not exchanged.
        - If the input frame is of type "Grayscale" and "debayer_pattern" is "RGB" or "BGR", the result
        is a three-channel RGB / BGR image where all channels are the same.
        - If a non-standard "debayer_pattern" (i.e. "RGGB", "GRBG", "GBRG", "BGGR") is specified and the
        input is a B/W image, decode the image using the given Bayer pattern. If the input image is
        of type "color" (three-channel RGB or BGR), first convert it into grayscale and then decode
        the image as in the B/W case. In both cases the result is a three-channel RGB color image.

        :param frame_in: Input image, either 2D (grayscale) or 3D (color). The type is either 8 or 16
                        bit unsigned int.
        :param debayer_pattern: Pattern used to convert the input image into the output image. One out
                                of 'Grayscale', 'RGB', 'Force Bayer RGGB', 'Force Bayer GRBG',
                                'Force Bayer GBRG', 'Force Bayer BGGR'
        :param BGR_input: If 'True', a color input frame is interpreted as 'BGR'; Otherwise as 'RGB'.
                        OpenCV reads color images in 'BGR' format.
        :return: frame_out: output image (see above)
        '''
        if debayer_method == 'Bilinear':
            debayer_codes = {
                'Force Bayer RGGB': COLOR_BayerRG2BGR,
                'Force Bayer GRBG': COLOR_BayerGR2BGR,
                'Force Bayer GBRG': COLOR_BayerGB2BGR,
                'Force Bayer BGGR': COLOR_BayerBG2BGR
            }
        if debayer_method == 'Variable Number of Gradients':
            debayer_codes = {
                'Force Bayer RGGB': COLOR_BayerRG2BGR_VNG,
                'Force Bayer GRBG': COLOR_BayerGR2BGR_VNG,
                'Force Bayer GBRG': COLOR_BayerGB2BGR_VNG,
                'Force Bayer BGGR': COLOR_BayerBG2BGR_VNG
            }
        if debayer_method == 'Edge Aware':
            debayer_codes = {
                'Force Bayer RGGB': COLOR_BayerRG2BGR_EA,
                'Force Bayer GRBG': COLOR_BayerGR2BGR_EA,
                'Force Bayer GBRG': COLOR_BayerGB2BGR_EA,
                'Force Bayer BGGR': COLOR_BayerBG2BGR_EA
            }

        type_in = frame_in.dtype

        if type_in != uint8 and type_in != uint16:
            raise Exception("Image type " + str(type_in) + " not supported")

        # If the input frame is 3D, it represents a color image.
        color_in = len(frame_in.shape) == 3

        # Case color input image.
        if color_in:
            # Three-channel input, interpret as RGB color and leave it unchanged.
            if debayer_pattern == 'No change' or debayer_pattern == 'RGB' and not BGR_input or \
                    debayer_pattern == 'BGR' and BGR_input:
                frame_out = frame_in

            # If the Bayer pattern and the BGR_input flag don't match, flip channels.
            elif debayer_pattern == 'RGB' and BGR_input or debayer_pattern == 'BGR' and not BGR_input:
                frame_out = cvtColor(frame_in, COLOR_BGR2RGB)

            # Three-channel (color) input, reduce to two-channel (B/W) image.
            elif debayer_pattern in ['Grayscale', 'Force Bayer RGGB', 'Force Bayer GRBG',
                                    'Force Bayer GBRG', 'Force Bayer BGGR']:

                frame_2D = cvtColor(frame_in, COLOR_RGB2GRAY)

                # Output is B/W image.
                if debayer_pattern == 'Grayscale':
                    frame_out = frame_2D

                # Decode the B/W image into a color image using a Bayer pattern.
                else:
                    frame_out = cvtColor(frame_2D, debayer_codes[debayer_pattern])

            # Invalid debayer pattern specified.
            else:
                raise Exception("Debayer pattern " + debayer_pattern + " not supported")

        # Case B/W input image.
        else:
            # Two-channel input, interpret as B/W image and leave it unchanged.
            if debayer_pattern in ['No change', 'Grayscale']:
                frame_out = frame_in

            # Transform the one-channel B/W image in an RGB one where all three channels are the same.
            elif debayer_pattern == 'RGB' or debayer_pattern == 'BGR':
                frame_out = cvtColor(frame_in, COLOR_GRAY2RGB)

            # Non-standard Bayer pattern, decode into color image.
            elif debayer_pattern in ['Force Bayer RGGB', 'Force Bayer GRBG',
                                    'Force Bayer GBRG', 'Force Bayer BGGR']:
                frame_out = cvtColor(frame_in, debayer_codes[debayer_pattern])

            # Invalid Bayer pattern specified.
            else:
                raise Exception("Debayer pattern " + debayer_pattern + " not supported")

        # Return the decoded image.
        return frame_out


    def detect_bayer(frame, frames_bayer_max_noise_diff_green, frames_bayer_min_distance_from_blue,
                    frames_color_difference_threshold):
        '''
        Detect a Bayer pattern in a grayscale image. The assumption is that statistically the
        brightness differences at adjacent pixels are greater than those at neighboring pixels of the
        same color. It is also assumed that the noise in the blue channel is greater than in the red
        and green channels.

        Acknowledgements: This method uses an algorithm developed by Chris Garry for his 'PIPP'
                        software package.

        :param frame: Numpy array (2D or 3D) of type uint8 or uint16 containing the image data.
        :param frames_bayer_max_noise_diff_green: Maximum allowed difference in noise levels at the two
                                                green pixels of the bayer matrix in percent of value
                                                at the blue pixel.
        :param frames_bayer_min_distance_from_blue: Maximum allowed noise level at Bayer matrix pixels
                                                    other than blue, in percent of value at the blue
                                                    pixel.
        :param frames_color_difference_threshold: If the brightness values of all three color channels
                                                of a three-channel frame at all pixels do not differ
                                                by more than this value, the frame is regarded as
                                                monochrome.

        :return: If the input frame is a (3D) color image, 'Color' is returned.
                If the input frame is a (2D or 3D) grayscale image without any detectable Bayer
                pattern, 'Grayscale' is returned.
                If a Bayer pattern is detected in the grayscale image, its type (one out of
                'Force Bayer BGGR', 'Force Bayer GBRG', 'Force Bayer GRBG', 'Force Bayer RGGB') is
                returned.
                If none of the above is true, 'None' is returned.
        '''

        # Frames are stored as 3D arrays. Test if all three color levels are the same.
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            first_minus_second = np_max(
                abs(frame[:, :, 0].astype(int32) - frame[:, :, 1].astype(int32)))
            first_minus_third = np_max(abs(frame[:, :, 0].astype(int32) - frame[:, :, 2].astype(int32)))
            # print("first_minus_second: " + str(first_minus_second) + ", first_minus_third: " + str(
            #     first_minus_third))

            # The color levels differ more than some threshold. Probably color image.
            if first_minus_second > frames_color_difference_threshold \
                    or first_minus_third > frames_color_difference_threshold:
                return 'Color'

            # All three levels are the same, convert to 2D grayscale image and proceed.
            else:
                frame_grayscale = cvtColor(frame, COLOR_RGB2GRAY).astype(int32)

        # Input frame is already grayscale.
        elif len(frame.shape) == 2:
            frame_grayscale = frame.astype(int32)

        # Neither a color image nor grayscale.
        else:
            return 'None'

        try:
            height, width = frame_grayscale.shape
            analysis_height = height - height % 2
            analysis_width = width - width % 2

            # Look for signs of a bayer pattern.
            adjacent_pixel_diffs = np_sum(
                abs(frame_grayscale[:, 0:analysis_width - 2] - frame_grayscale[:,
                                                            1:analysis_width - 1]))
            apart_pixel_diffs = np_sum(
                abs(frame_grayscale[:, 0:analysis_width - 2] - frame_grayscale[:, 2:analysis_width]))

            # Pixels are more like the pixels next to them than they are like the pixels 2 pixel away.
            # This indicates that there is no bayer pattern present
            if apart_pixel_diffs > adjacent_pixel_diffs:
                return 'Grayscale'

            # Analyse noise characteristics of image to guess at positions of R, G and B in bayer pattern.
            noise_level = ndarray((2, 2), dtype=float)
            for y in range(2):
                for x in range(2):
                    # Apply a five point (Poisson) stencil and sum over all points.
                    neighbors = (frame_grayscale[y:analysis_height - 6 + y:2,
                                2 + x:analysis_width - 4 + x:2] +
                                frame_grayscale[4 + y:analysis_height - 2 + y:2,
                                2 + x:analysis_width - 4 + x:2] +
                                frame_grayscale[2 + y:analysis_height - 4 + y:2,
                                x:analysis_width - 6 + x:2] +
                                frame_grayscale[2 + y:analysis_height - 4 + y:2,
                                4 + x:analysis_width - 2 + x:2]) / 4.
                    noise_level[y, x] = np_sum(
                        abs(frame_grayscale[2 + y:analysis_height - 4 + y:2,
                            2 + x:analysis_width - 4 + x:2] - neighbors))

            # Normalize noise levels.
            max_y, max_x = unravel_index(noise_level.argmax(), noise_level.shape)
            max_noise_level = noise_level[max_y, max_x]
            if max_noise_level > 0.:
                noise_level = noise_level * (100. / max_noise_level)
            # Zero noise - cannot detect bayer pattern.
            else:
                return 'Grayscale'

            # The location of the maximum noise level is interpreted as the blue channel.
            # It is in position (0, 0).
            if (max_y, max_x) == (0, 0):
                # The noise levels of the green pixels are too different for this to be a bayer pattern.
                if abs(noise_level[0, 1] - noise_level[1, 0]) > frames_bayer_max_noise_diff_green:
                    return 'Grayscale'
                # Noise levels of the other pixels are too close to the blue values for this to definitely
                # be a bayer pattern.
                if noise_level[0, 1] > frames_bayer_min_distance_from_blue or noise_level[
                    1, 0] > frames_bayer_min_distance_from_blue or noise_level[
                    1, 1] > frames_bayer_min_distance_from_blue:
                    return 'Grayscale'
                # Bayer pattern "BGGR" found.
                return 'Force Bayer BGGR'

            # Case "GBRG":
            elif (max_y, max_x) == (0, 1):
                if abs(noise_level[0, 0] - noise_level[1, 1]) > frames_bayer_max_noise_diff_green:
                    return 'Grayscale'
                if noise_level[0, 0] > frames_bayer_min_distance_from_blue or noise_level[
                    1, 0] > frames_bayer_min_distance_from_blue or noise_level[
                    1, 1] > frames_bayer_min_distance_from_blue:
                    return 'Grayscale'
                # Bayer pattern "GBRG" found.
                return 'Force Bayer GBRG'

            # Case "GRBG":
            elif (max_y, max_x) == (1, 0):
                if abs(noise_level[0, 0] - noise_level[1, 1]) > frames_bayer_max_noise_diff_green:
                    return 'Grayscale'
                if noise_level[0, 0] > frames_bayer_min_distance_from_blue or noise_level[
                    0, 1] > frames_bayer_min_distance_from_blue or noise_level[
                    1, 1] > frames_bayer_min_distance_from_blue:
                    return 'Grayscale'
                # Bayer pattern "GRBG" found.
                return 'Force Bayer GRBG'

            # Case "RGGB"
            elif (max_y, max_x) == (1, 1):
                if abs(noise_level[0, 1] - noise_level[1, 0]) > frames_bayer_max_noise_diff_green:
                    return 'Grayscale'
                if noise_level[0, 0] > frames_bayer_min_distance_from_blue or noise_level[
                    0, 1] > frames_bayer_min_distance_from_blue or noise_level[
                    1, 0] > frames_bayer_min_distance_from_blue:
                    return 'Grayscale'
                # Bayer pattern "RGGB" found.
                return 'Force Bayer RGGB'

            # Theoretically this cannot be reached.
            return 'None'

        # If something bad has happened, return 'None'.
        except Exception as e:
            # print("An Exception occurred: " + str(e))
            return 'None'

    def detect_rgb_bgr(frame):
        """
        Given a color (3D) frame, find out if the channels are arranged as 'RGB' or 'BGR'. To this end,
        the noise level of the first and third channels are compared. The channel with the highest noise
        level is interpreted as the blue channel.

        :param frame: Numpy array (3D) of type uint8 or uint16 containing the image data.
        :return: Either 'RGB' or 'BGR', depending on whether the noise in the third or first channel is
                highest. If an error occurs, 'None' is returned.
        """

        # Not a color (3D) frame:
        if len(frame.shape) != 3:
            return 'None'

        try:
            height, width = frame.shape[0:2]
            analysis_height = height - height % 2
            analysis_width = width - width % 2

            # Analyse noise characteristics of image to guess at positions of R, G and B in bayer pattern.
            frame_int32 = frame.astype(int32)
            noise_level = [0., 0., 0.]
            for channel in [0, 2]:
                # Apply a five point (Poisson) stencil and sum over all points.
                neighbors = (frame_int32[0:analysis_height - 2, 1:analysis_width - 1, channel] +
                            frame_int32[2:analysis_height, 1:analysis_width - 1, channel] +
                            frame_int32[1:analysis_height - 1, 0:analysis_width - 2, channel] +
                            frame_int32[1:analysis_height - 1, 2:analysis_width, channel]) / 4.
                noise_level[channel] = np_sum(
                    abs(frame_int32[1:analysis_height - 1, 1:analysis_width - 1, channel] - neighbors))

            # print("noise level 0:" + str(noise_level[0]) + ", noise level 2:" + str(noise_level[2]))
            if noise_level[0] > noise_level[2]:
                return 'BGR'
            else:
                return 'RGB'
        except Exception as e:
            # print("An Exception occurred: " + str(e))
            return 'None'
        


    def save_image(filename, image, color=False, avoid_overwriting=True,
                   header="PlanetarySystemStacker"):
        """
        Save an image to a file. If "avoid_overwriting" is set to False, images can have either
        ".png", ".tiff" or ".fits" format.

        :param filename: Name of the file where the image is to be written
        :param image: ndarray object containing the image data
        :param color: If True, a three channel RGB image is to be saved. Otherwise, it is assumed
                      that the image is monochrome.
        :param avoid_overwriting: If True, append a string to the input name if necessary so that
                                  it does not match any existing file. If False, overwrite
                                  an existing file.
        :param header: String with information on the PSS version being used (optional).
        :return: -
        """

        # Handle the special case of .fits files first.
        if Path(filename).suffix == '.fits':
            # Flip image horizontally to preserve orientation
            image = flip(image, 0)
            if color:
                image = moveaxis(image, -1, 0)
            hdu = fits.PrimaryHDU(image)
            hdu.header['CREATOR'] = header
            hdu.writeto(filename, overwrite=True)

        # Not a .fits file, the name can either point to a file or a directory, or be new.
        else:
            if Path(filename).is_dir():
                # If a directory with the given name already exists, append the word "_file".
                filename += '_file'
                if avoid_overwriting:
                    while True:
                        if not Path(filename + '.png').exists():
                            break
                        filename += '_file'
                    filename += '.png'
                else:
                    filename += '.png'
                    if Path(filename).is_file():
                        remove(filename)

            # It is a file.
            elif Path(filename).is_file():
                # If overwriting is to be avoided, try to append "_copy" to its basename.
                # If it still exists, repeat.
                if avoid_overwriting:
                    suffix = Path(filename).suffix
                    while True:
                        p = Path(filename)
                        filename = Path.joinpath(p.parents[0], p.stem + '_copy' + suffix)
                        if not Path(filename).exists():
                            break
                # File may be overwritten. Delete it first.
                else:
                    remove(filename)

            # It is a new name. If it does not have a file suffix, add the default '.png'.
            else:
                # If the file name is new and has no suffix, add ".png".
                if not Path(filename).suffix:
                    filename += '.png'

            if Path(filename).suffix not in ['.tiff', '.png']:
                raise TypeError("Attempt to write image format other than '.tiff' or '.png'")

            # Write the image to the file. Before writing, convert the internal RGB representation
            # into the BGR representation assumed by OpenCV.
            if color:
                imwrite(str(filename), cvtColor(image, COLOR_RGB2BGR))
            else:
                imwrite(str(filename), image)
