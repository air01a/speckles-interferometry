import numpy as np
import scipy as sp
from scipy import optimize as opt


class IDAN:
    """
    IDAN filter. Intensity driven adaptive neighborhood filter.

    further information:
    G. Vasile et al. “Intensity-Driven Adaptive-Neighborhood Technique for Polarimetric and Interferometric
    SAR Parameters Estimation” IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING,Vol 44. No. 6, 2006


    :author: Joel Amao
    :param array: The image to filter (2D np.ndarray)
    :type array: float
    :param looks = 1.0: The effective number of looks of the input image.
    :param nmax = 50: The maximum number of neighbours to add
    :param LLMMSE: Activates LLMMSE processing after the region growing
    :type looks: float
    :returns: filtered image
    """
    gui = {'menu': 'SAR|Speckle filter', 'entry': 'IDAN filter'}
    para = [
        {'var': 'looks', 'value': 1.0, 'type': 'float', 'range': [1.0, 99.0], 'text': '# of looks'},
        {'var': 'nmax', 'value': 50, 'type': 'int', 'range': [40, 200], 'text': 'Max # of Neighbors'},
        {'var': 'LLMMSE', 'value': True, 'type': 'bool', 'text': 'Activates LLMMSE processing'}
    ]


    def __init__(self, *args, **kwargs):
        self.name = "IDAN FILTER"
        self.nmax = 50
        self.looks = 1.0
        self.LLMMSE = True

        self.blockprocess = True
        self.blocksize = self.nmax * 4
        self.blockoverlap = self.nmax
        # self.nthreads = 1
        # todo: add a mininum nmax parameters
        # todo: test InSAR data

    def filter(self, array, *args, **kwargs):

        MMSEproc = self.LLMMSE
        array[np.isnan(array)] = 0.0
        shape = array.shape
        if len(shape) == 3:
            array = np.abs(array)
            span = np.sum(array ** 2, axis=0)
            array = array[np.newaxis, ...]
        elif len(shape) == 4:
            span = np.abs(np.trace(array, axis1=0, axis2=1))
        else:
            array = np.abs(array)
            span = array ** 2
            array = array[np.newaxis, np.newaxis, ...]

        win = np.array([3, 3])
        ldim = array.shape[0:2]
        rdim = array.shape[2:4]
        nmax = self.nmax
        nlook = 1 / (np.sqrt(self.looks)) / 3.0

        # ==============================================================================
        # 1.1 Rough estimation of the seed value
        # ==============================================================================

        intensities = np.zeros((ldim[0], rdim[0], rdim[1]), dtype=np.float32)
        med_arr = np.zeros_like(intensities)
        for i in range(ldim[0]):
            intensities[i, :] = array[i, i, :, :].real
            med_arr[i, :] = sp.ndimage.filters.median_filter(intensities[i, :], size=win)

        data = array
        ldim = data.shape[0:2]
        rdim = data.shape[2:4]
        maxp = rdim[0] * rdim[1]

        mask_vec = np.zeros(shape=[maxp, nmax], dtype=np.int64)
        background = np.zeros_like(mask_vec)
        nnList = np.zeros((maxp, 1), dtype=np.int64)

        # k = (xx)*rdim[1] + (yy)

        # ==============================================================================
        # 1.2 Region growing
        # ==============================================================================

        thres = 2 * nlook
        intensities = intensities.reshape((ldim[0], maxp))
        med_arr = med_arr.reshape(ldim[0], maxp)
        neighbours = [1, -1, -rdim[1], rdim[1], rdim[1] + 1, rdim[1] - 1, -rdim[1] + 1, -rdim[1] - 1]

        for k in range(maxp):
            stack = [(k)]
            nn = 0
            bg_nn = 0
            while stack:
                y = stack.pop(0)
                nnList[k] = nn
                if nn == 0:
                    mask_vec[k, 0] = y

                for dy in neighbours:
                    ny = y + dy
                    # and >>> and
                    if (not (ny in background[k, :]) and 0 <= ny and not (
                        ny in mask_vec[k, :]) and nn < nmax and ny < maxp):

                        up = np.abs(intensities[:, ny] - med_arr[:, k])
                        down = np.maximum(np.abs(med_arr[:, k]), 1e-10)
                        sum = (up / down).sum()

                        if (sum <= thres):
                            stack.append(ny)
                            if nn < nmax:
                                nn += 1
                            if nn < nmax:
                                mask_vec[k, nn] = ny
                        else:
                            if bg_nn < nmax:
                                background[k, bg_nn] = ny
                                bg_nn += 1

        # ==============================================================================
        # 2.1 Refined estimation of the seed value
        # ==============================================================================
        dataV = data.reshape(ldim[0], ldim[1], maxp)
        updatedSeed = np.zeros_like(med_arr)

        for k in range(maxp):
            for i in range(ldim[0]):
                if np.where(mask_vec[k, :] > 0)[0].size > 0:
                    updatedSeed[i, k] = np.mean(dataV[i, i, mask_vec[k, :][np.where(mask_vec[k, :] > 0)]].real)

        # ==============================================================================
        # 2.2 Reinspection of the background pixels
        # ==============================================================================
        thres = 6 * nlook

        for k in range(maxp):
            if nnList[k] < nmax:
                stack = background[k][np.where(background[k] > 0)].tolist()
                while stack:
                    nnList[k] += 1
                    y = stack.pop(0)
                    up = np.abs(intensities[:, y] - updatedSeed[:, k])
                    down = np.maximum(np.abs(updatedSeed[:, k]), 1e-10)
                    sum = (up / down).sum()
                    if (sum <= thres):
                        if nnList[k] < nmax:
                            mask_vec[k, nnList[k]] = y

        # ==============================================================================
        # 3.1 Parameter estimation
        # ==============================================================================
        avg = np.zeros(shape=[ldim[0], ldim[1], maxp], dtype=np.complex64)

        for k in range(maxp):
            for i in range(ldim[0]):
                for j in range(ldim[1]):
                    if np.where(mask_vec[k, :] > 0)[0].size > 0:
                        idx = mask_vec[k, :][np.where(mask_vec[k, :] > 0)]
                        avg[i, j, k] = np.mean(dataV[i, j, idx])
                    else:
                        avg[i, j, k] = dataV[i, j, k]

        # Calculates the squared mean of the array (marr) and the mean squared array (m2arr)
        m2arr = sp.ndimage.filters.uniform_filter(span ** 2, size=win)
        marr = sp.ndimage.filters.uniform_filter(span, size=win)
        # Variance within window, follows the identity Var(x) = E(x**2) - [E(X)]**2
        vary = (m2arr - marr ** 2).clip(1e-10)
        # Standard deviation within window
        stdDev = np.sqrt(vary)
        # cu and ci are the main parameters of the weight function w
        cu = np.sqrt(1 / self.looks)
        ci = (stdDev / marr)

        # Clipped weighted function
        w = (1 - ((cu ** 2) / (ci ** 2))).clip(0) / ((1 + (cu ** 2)).clip(1e-10))
        w = w.reshape(maxp)

        # LLMMSE
        if MMSEproc:
            LLMMSE = np.zeros_like(avg, dtype=np.complex64)
            for k in range(maxp):
                for i in range(ldim[0]):
                    for j in range(ldim[1]):
                        LLMMSE[i, j, k] = avg[i, j, k] + w[k] * (dataV[i, j, k] - avg[i, j, k])
            LLMMSE = LLMMSE.reshape(ldim[0], ldim[1], rdim[0], rdim[1])
            return np.squeeze(LLMMSE)
        else:
            # Reshaping the array and removing the borders
            avg = avg.reshape(ldim[0], ldim[1], rdim[0], rdim[1])
            return np.squeeze(avg)


def idan(*args, **kwargs):
    return IDAN(*args, **kwargs).filter(*args, **kwargs)