from cv2 import GaussianBlur, cvtColor, COLOR_BGR2HSV, COLOR_HSV2BGR, BORDER_DEFAULT, \
    resize
from numpy import zeros, empty,  float32, uint16, percentile, clip, uint8,interp

from math import exp
I16_BITS_MAX_VALUE = 65535

class AstroImageProcessing:
    @staticmethod
    def levels(image, blacks: float, midtones: float, whites: float, contrast: float =1, r:float =1, g:float=1, b:float=1):
        min = image.min()
        max = image.max()
        median = max - min
        if midtones <= 0:
            midtones = 0.1
        # midtones
        image = I16_BITS_MAX_VALUE *((((image-min)/median - 0.5) * contrast + 0.5)  * median + min ) ** (1 / midtones) / I16_BITS_MAX_VALUE ** (1 / midtones)
        #black / white levels
        image = clip(image, blacks, whites)

        if (len(image.shape)<3):
            image = float32(interp(image,
                                                (min, max),
                                                (0, I16_BITS_MAX_VALUE)))
        else:
            image[:,:,0] = float32(interp(image[:,:,0],
                                                (min, max),
                                                (0, I16_BITS_MAX_VALUE)))

        if (len(image.shape)>2):
            image[:,:0] = image[:,:0] * r
            image[:,:1] = image[:,:1]* g
            image[:,:2] = image[:,:2] * b


        image = clip(image.data, 0, 2**16 - 1)
        return image

    @staticmethod
    def wavelet_sharpen(input_image, amount, radius):
        """
        Sharpen a B/W or color image with wavelets. The underlying algorithm was taken from the
        Gimp wavelet plugin, originally written in C and published under the GPLv2+ license at:
        https://github.com/mrossini-ethz/gimp-wavelet-sharpen/blob/master/src/wavelet.c

        :param input_image: Input image (B/W or color), type uint16
        :param amount: Amount of sharpening
        :param radius: Radius in pixels
        :return: Sharpened image, same format as input image
        """

        height, width = input_image.shape[:2]
        color = len(input_image.shape) == 3

        # Allocate workspace: Three complete images, plus 1D object with length max(row, column).
        if color:
            fimg = empty((3, height, width, 3), dtype=float32)
            temp = zeros((max(width, height), 3), dtype=float32)
        else:
            fimg = empty((3, height, width), dtype=float32)
            temp = zeros(max(width, height), dtype=float32)

        # Convert input image to floats.
        fimg[0] = input_image / 65535

        # Start with level 0. Store its Laplacian on level 1. The operator is separated in a
        # column and a row operator.
        hpass = 0
        for lev in range(5):
            # Highpass and lowpass levels use image indices 1 and 2 in alternating mode to save
            # space.
            lpass = ((lev & 1) + 1)

            if color:
                for row in range(height):
                    AstroImageProcessing.mexican_hat_color(temp, fimg[hpass][row, :, :], width, 1 << lev)
                    fimg[lpass][row, :, :] = temp[:width, :] * 0.25
                for col in range(width):
                    AstroImageProcessing.mexican_hat_color(temp, fimg[lpass][:, col, :], height, 1 << lev)
                    fimg[lpass][:, col, :] = temp[:height, :] * 0.25
            else:
                for row in range(height):
                    AstroImageProcessing.mexican_hat(temp, fimg[hpass][row, :], width, 1 << lev)
                    fimg[lpass][row, :] = temp[:width] * 0.25
                for col in range(width):
                    AstroImageProcessing.mexican_hat(temp, fimg[lpass][:, col], height, 1 << lev)
                    fimg[lpass][:, col] = temp[:height] * 0.25

            # Compute the amount of the correction at the current level.
            amt = amount * exp(-(lev - radius) * (lev - radius) / 1.5) + 1.

            fimg[hpass] -= fimg[lpass]
            fimg[hpass] *= amt

            # Accumulate all corrections in the first workspace image.
            if hpass:
                fimg[0] += fimg[hpass]

            hpass = lpass

        # At the end add the coarsest level and convert back to 16bit integer format.
        fimg[0] = ((fimg[0] + fimg[lpass]) * 65535.).clip(min=0., max=65535.)
        return fimg[0].astype(uint16)

    @staticmethod
    def mexican_hat(temp, base, size, sc):
        """
        Apply a 1D strided second derivative to a row or column of a B/W image. Store the result
        in the temporary workspace "temp".

        :param temp: Workspace (type float32), length at least "size" elements
        :param base: Input image (B/W), Type float32
        :param size: Length of image row / column
        :param sc: Stride (power of 2) of operator
        :return: -
        """

        # Special case at begin of row/column. Full operator not applicable.
        temp[:sc] = 2 * base[:sc] + base[sc:0:-1] + base[sc:2 * sc]
        # Apply the full operator.
        temp[sc:size - sc] = 2 * base[sc:size - sc] + base[:size - 2 * sc] + base[2 * sc:size]
        # Special case at end of row/column. The full operator is not applicable.
        temp[size - sc:size] = 2 * base[size - sc:size] + base[size - 2 * sc:size - sc] + \
                               base[size - 2:size - 2 - sc:-1]

    @staticmethod
    def mexican_hat_color(temp, base, size, sc):
        """
        Apply a 1D strided second derivative to a row or column of a color image. Store the result
        in the temporary workspace "temp".

        :param temp: Workspace (type float32), length at least "size" elements (first dimension)
                     times 3 colors (second dimension).
        :param base: Input image (color), Type float32
        :param size: Length of image row / column
        :param sc: Stride (power of 2) of operator
        :return: -
        """

        # Special case at begin of row/column. Full operator not applicable.
        temp[:sc, :] = 2 * base[:sc, :] + base[sc:0:-1, :] + base[sc:2 * sc, :]
        # Apply the full operator.
        temp[sc:size - sc, :] = 2 * base[sc:size - sc, :] + base[:size - 2 * sc, :] + base[
                                2 * sc:size, :]
        # Special case at end of row/column. The full operator is not applicable.
        temp[size - sc:size, :] = 2 * base[size - sc:size, :] + base[size - 2 * sc:size - sc, :] + \
                                  base[size - 2:size - 2 - sc:-1, :]


    @staticmethod
    def gaussian_sharpen(input_image, amount, radius, luminance_only=False):
        """
        Sharpen an image with a Gaussian kernel. The input image can be B/W or color.

        :param input_image: Input image, type uint16
        :param amount: Amount of sharpening
        :param radius: Radius of Gaussian kernel (in pixels)
        :param luminance_only: True, if only the luminance channel of a color image is to be
                               sharpened. Default is False.
        :return: The sharpened image (B/W or color, as input), type uint16
        """

        color = len(input_image.shape) == 3

        # Translate the kernel radius into standard deviation.
        sigma = radius / 3

        # Convert the image to floating point format.
        image = input_image.astype(float32)

        # Special case: Only sharpen the luminance channel of a color image.
        if color and luminance_only:
            hsv = cvtColor(image, COLOR_BGR2HSV)
            luminance = hsv[:, :, 2]

            # Apply a Gaussian blur filter, subtract it from the original image, and add a multiple
            # of this correction to the original image. Clip values out of range.
            luminance_blurred = GaussianBlur(luminance, (0, 0), sigma, borderType=BORDER_DEFAULT)
            hsv[:, :, 2] = (luminance + amount * (luminance - luminance_blurred)).clip(min=0.,
                                                                                       max=65535.)
            # Convert the image back to uint16.
            return cvtColor(hsv, COLOR_HSV2BGR).astype(uint16)
        # General case: Treat the entire image (B/W or color 16bit mode).
        else:
            image_blurred = GaussianBlur(image, (0, 0), sigma, borderType=BORDER_DEFAULT)
            return (image + amount * (image - image_blurred)).clip(min=0., max=65535.).astype(
                uint16)

    @staticmethod
    def gaussian_blur(input_image, amount, radius, luminance_only=False):
        """
        Soften an image with a Gaussian kernel. The input image can be B/W or color.

        :param input_image: Input image, type uint16
        :param amount: Amount of blurring, between 0. and 1.
        :param radius: Radius of Gaussian kernel (in pixels)
        :param luminance_only: True, if only the luminance channel of a color image is to be
                               blurred. Default is False.
        :return: The blurred image (B/W or color, as input), type uint16
        """

        color = len(input_image.shape) == 3

        # Translate the kernel radius into standard deviation.
        sigma = radius / 3

        # Convert the image to floating point format.
        image = input_image.astype(float32)

        # Special case: Only blur the luminance channel of a color image.
        if color and luminance_only:
            hsv = cvtColor(image, COLOR_BGR2HSV)
            luminance = hsv[:, :, 2]

            # Apply a Gaussian blur filter, subtract it from the original image, and add a multiple
            # of this correction to the original image. Clip values out of range.
            luminance_blurred = GaussianBlur(luminance, (0, 0), sigma, borderType=BORDER_DEFAULT)
            hsv[:, :, 2] = (luminance_blurred*amount + luminance*(1.-amount)).clip(min=0.,
                                                                                       max=65535.)
            # Convert the image back to uint16.
            return cvtColor(hsv, COLOR_HSV2BGR).astype(uint16)
        # General case: Treat the entire image (B/W or color 16bit mode).
        else:
            image_blurred = GaussianBlur(image, (0, 0), sigma, borderType=BORDER_DEFAULT)
            return (image_blurred*amount + image*(1.-amount)).clip(min=0., max=65535.).astype(
                uint16)

    
    @staticmethod
    def stretch(image, intensity=0.2):
        
        min_val = percentile(image, intensity)
        max_val = percentile(image, 100 - intensity)

        if (min_val!=max_val):
            image = (clip((image - min_val) * (65535.0 / (max_val - min_val) ), 0, 65535)).astype(uint16)

        return image

    @staticmethod
    def image_resize_width(image, new_width):
        h = int(image.shape[0]*new_width/image.shape[1])
        return resize(image, (new_width, h))
    
    @staticmethod
    def image_resize(image, factor):
        return resize(image, (int(image.shape[1]*factor), int(image.shape[0]*factor)))


if __name__=="__main__":
    from cv2 import imshow, waitKey,destroyAllWindows
    from imagemanager import ImageManager
    from qualitytest import QualityTest

    frame = ImageManager.read_image('test/finam.jpg') #'test/15_44_11_sun_lapl5_ap88.tif')
    frame = AstroImageProcessing.stretch(AstroImageProcessing.image_resize_width(frame, 640))

    print("init",QualityTest.local_contrast_laplace(frame))
    initial = frame.copy()
    imshow('test',frame)
    waitKey(0)
    
    frame = AstroImageProcessing.wavelet_sharpen(frame, 1.4,10)
    print("wavelet",QualityTest.local_contrast_laplace(frame))
    imshow('test1',frame)
    waitKey(0)

    frame = initial.copy()
    frame = AstroImageProcessing.gaussian_sharpen(frame, 10,3)
    print("Gaussian",QualityTest.local_contrast_laplace(frame))
    imshow('test2',frame)
    waitKey(0)
    ImageManager.save_image('test.png', frame)

    frame = initial.copy()
    frame = AstroImageProcessing.gaussian_blur(frame, 40,1)
    print("Blur",QualityTest.local_contrast_laplace(frame))
    imshow('test3',frame)
    waitKey(0)
    frame = AstroImageProcessing.gaussian_sharpen(frame, 10,3)
    print("Blur + gaussian",QualityTest.local_contrast_laplace(frame))
    imshow('test4',frame)
    waitKey(0)



    destroyAllWindows()