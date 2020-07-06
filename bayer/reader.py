#!/usr/bin/env python3

"""
DDFAPD - Menon (2007) Bayer CFA Demosaicing
===========================================

*Bayer* CFA (Colour Filter Array) DDFAPD - *Menon (2007)* demosaicing.

References
----------
-   :cite:`Menon2007c` : Menon, D., Andriani, S., & Calvagno, G. (2007).
    Demosaicing With Directional Filtering and a posteriori Decision. IEEE
    Transactions on Image Processing, 16(1), 132-141.
    doi:10.1109/TIP.2006.884928
"""

import cv2
import numpy as np
from scipy.ndimage.filters import convolve, convolve1d

# from colour.utilities import as_float_array, tsplit, tstack

def masks_CFA_Bayer(shape, pattern='RGGB'):
    """
    Returns the *Bayer* CFA red, green and blue masks for given pattern.

    Parameters
    ----------
    shape : array_like
        Dimensions of the *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.

    Returns
    -------
    tuple
        *Bayer* CFA red, green and blue masks.

    Examples
    --------
    >>> from pprint import pprint
    >>> shape = (3, 3)
    >>> pprint(masks_CFA_Bayer(shape))
    (array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool))
    >>> pprint(masks_CFA_Bayer(shape, 'BGGR'))
    (array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool))
    """

    pattern = pattern.upper()

    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RGB')

def mosaicing_CFA_Bayer(RGB, pattern='RGGB'):
    """
    Returns the *Bayer* CFA mosaic for a given *RGB* colourspace array.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.

    Returns
    -------
    ndarray
        *Bayer* CFA mosaic.

    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.array([[[0, 1, 2],
    ...                  [0, 1, 2]],
    ...                 [[0, 1, 2],
    ...                  [0, 1, 2]]])
    >>> mosaicing_CFA_Bayer(RGB)
    array([[ 0.,  1.],
           [ 1.,  2.]])
    >>> mosaicing_CFA_Bayer(RGB, pattern='BGGR')
    array([[ 2.,  1.],
           [ 1.,  0.]])
    """

    B, G, R = cv2.split(RGB)
    R_m, G_m, B_m = masks_CFA_Bayer(RGB.shape[0:2], pattern)

    CFA = R * R_m + G * G_m + B * B_m

    return CFA

def _cnv_h(x, y):
    """
    Helper function for horizontal convolution.
    """

    return convolve1d(x, y, mode='mirror')


def _cnv_v(x, y):
    """
    Helper function for vertical convolution.
    """

    return convolve1d(x, y, mode='mirror', axis=0)


def demosaicing_CFA_Bayer_Menon2007(CFA, pattern='RGGB', refining_step=True):
    """
    Returns the demosaiced *RGB* colourspace array from given *Bayer* CFA using
    DDFAPD - *Menon (2007)* demosaicing algorithm.

    Parameters
    ----------
    CFA : array_like
        *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.
    refining_step : bool
        Perform refining step.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   The definition output is not clipped in range [0, 1] : this allows for
        direct HDRI / radiance image generation on *Bayer* CFA data and post
        demosaicing of the high dynamic range data as showcased in this
        `Jupyter Notebook <https://github.com/colour-science/colour-hdri/\
blob/develop/colour_hdri/examples/\
examples_merge_from_raw_files_with_post_demosaicing.ipynb>`__.

    References
    ----------
    :cite:`Menon2007c`

    Examples
    --------
    >>> CFA = np.array(
    ...     [[ 0.30980393,  0.36078432,  0.30588236,  0.3764706 ],
    ...      [ 0.35686275,  0.39607844,  0.36078432,  0.40000001]])
    >>> demosaicing_CFA_Bayer_Menon2007(CFA)
    array([[[ 0.30980393,  0.35686275,  0.39215687],
            [ 0.30980393,  0.36078432,  0.39607844],
            [ 0.30588236,  0.36078432,  0.39019608],
            [ 0.32156864,  0.3764706 ,  0.40000001]],
    <BLANKLINE>
           [[ 0.30980393,  0.35686275,  0.39215687],
            [ 0.30980393,  0.36078432,  0.39607844],
            [ 0.30588236,  0.36078432,  0.39019609],
            [ 0.32156864,  0.3764706 ,  0.40000001]]])
    >>> CFA = np.array(
    ...     [[ 0.3764706 ,  0.36078432,  0.40784314,  0.3764706 ],
    ...      [ 0.35686275,  0.30980393,  0.36078432,  0.29803923]])
    >>> demosaicing_CFA_Bayer_Menon2007(CFA, 'BGGR')
    array([[[ 0.30588236,  0.35686275,  0.3764706 ],
            [ 0.30980393,  0.36078432,  0.39411766],
            [ 0.29607844,  0.36078432,  0.40784314],
            [ 0.29803923,  0.3764706 ,  0.42352942]],
    <BLANKLINE>
           [[ 0.30588236,  0.35686275,  0.3764706 ],
            [ 0.30980393,  0.36078432,  0.39411766],
            [ 0.29607844,  0.36078432,  0.40784314],
            [ 0.29803923,  0.3764706 ,  0.42352942]]])
    """

    # convert those pixels to normalized float
    CFA = CFA.astype(np.float32) / 255.0
    # get the mosaic mask of the givin pattern
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    h_0 = np.array([0, 0.5, 0, 0.5, 0])
    h_1 = np.array([-0.25, 0, 0.5, 0, -0.25])

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    # firstly, re-construct Green Image of Horizontal & Vertical direction
    G_H = np.where(G_m == False, _cnv_h(CFA, h_0) + _cnv_h(CFA, h_1), G)
    G_V = np.where(G_m == False, _cnv_v(CFA, h_0) + _cnv_v(CFA, h_1), G)
    # 计算垂直方向色度， chromance
    C_H = np.where(R_m == 1, R - G_H, 0)
    C_H = np.where(B_m == 1, B - G_H, C_H)
    # 计算水平方向色度
    C_V = np.where(R_m == 1, R - G_V, 0)
    C_V = np.where(B_m == 1, B - G_V, C_V)
    # 镜像模式在数组两边填充，水平方向上做差，间距为2
    D_H = np.abs(C_H - np.pad(C_H, ((0, 0),
                                    (0, 2)), mode=str('reflect'))[:, 2:])
    # 在垂直方向做差，间距为2
    D_V = np.abs(C_V - np.pad(C_V, ((0, 2),
                                    (0, 0)), mode=str('reflect'))[2:, :])
    # 梯度计算矩阵
    k = np.array(
        [[0, 0, 1, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 3, 0, 3],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 1]])  # yapf: disable
    # 计算水平和垂直方向上的梯度
    d_H = convolve(D_H, k, mode='constant')
    d_V = convolve(D_V, np.transpose(k), mode='constant')

    mask = d_V >= d_H
    G = np.where(mask, G_H, G_V)    # 根据梯度大小决策绿色分量
    M = np.where(mask, 1, 0)        # 决策掩码矩阵
    print(M)
    # Red rows. np.newaxis, 将数组提升一个维度
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)

    print(np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]))
    print(R_r)

    k_b = np.array([0.5, 0, 0.5])

    R = np.where(
        np.logical_and(G_m == 1, R_r == 1),
        G + _cnv_h(R, k_b) - _cnv_h(G, k_b),
        R,
    )

    R = np.where(
        np.logical_and(G_m == 1, B_r == 1) == 1,
        G + _cnv_v(R, k_b) - _cnv_v(G, k_b),
        R,
    )

    B = np.where(
        np.logical_and(G_m == 1, B_r == 1),
        G + _cnv_h(B, k_b) - _cnv_h(G, k_b),
        B,
    )

    B = np.where(
        np.logical_and(G_m == 1, R_r == 1) == 1,
        G + _cnv_v(B, k_b) - _cnv_v(G, k_b),
        B,
    )
    # 决策
    R = np.where(
        np.logical_and(B_r == 1, B_m == 1),
        np.where(
            M == 1,
            B + _cnv_h(R, k_b) - _cnv_h(B, k_b),
            B + _cnv_v(R, k_b) - _cnv_v(B, k_b),
        ),
        R,
    )

    B = np.where(
        np.logical_and(R_r == 1, R_m == 1),
        np.where(
            M == 1,
            R + _cnv_h(B, k_b) - _cnv_h(R, k_b),
            R + _cnv_v(B, k_b) - _cnv_v(R, k_b),
        ),
        B,
    )

    BGR = cv2.merge([B, G, R])   # 合并R、G、B分量, opencv使用bgr分量顺序

    if refining_step:
        # RGB = refining_step_Menon2007(BGR, cv2.merge([B_m, R_m, G_m]), M)
        pass

    return BGR

demosaicing_CFA_Bayer_DDFAPD = demosaicing_CFA_Bayer_Menon2007

def refining_step_Menon2007(RGB, RGB_m, M):
    """
    Performs the refining step on given *RGB* colourspace array.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    RGB_m : array_like
        *Bayer* CFA red, green and blue masks.
    M : array_like
        Estimation for the best directional reconstruction.

    Returns
    -------
    ndarray
        Refined *RGB* colourspace array.

    Examples
    --------
    >>> RGB = np.array(
    ...     [[[0.30588236, 0.35686275, 0.3764706],
    ...       [0.30980393, 0.36078432, 0.39411766],
    ...       [0.29607844, 0.36078432, 0.40784314],
    ...       [0.29803923, 0.37647060, 0.42352942]],
    ...      [[0.30588236, 0.35686275, 0.3764706],
    ...       [0.30980393, 0.36078432, 0.39411766],
    ...       [0.29607844, 0.36078432, 0.40784314],
    ...       [0.29803923, 0.37647060, 0.42352942]]])
    >>> RGB_m = np.array(
    ...     [[[0, 0, 1],
    ...       [0, 1, 0],
    ...       [0, 0, 1],
    ...       [0, 1, 0]],
    ...      [[0, 1, 0],
    ...       [1, 0, 0],
    ...       [0, 1, 0],
    ...       [1, 0, 0]]])
    >>> M = np.array(
    ...     [[0, 1, 0, 1],
    ...      [1, 0, 1, 0]])
    >>> refining_step_Menon2007(RGB, RGB_m, M)
    array([[[ 0.30588236,  0.35686275,  0.3764706 ],
            [ 0.30980393,  0.36078432,  0.39411765],
            [ 0.29607844,  0.36078432,  0.40784314],
            [ 0.29803923,  0.3764706 ,  0.42352942]],
    <BLANKLINE>
           [[ 0.30588236,  0.35686275,  0.3764706 ],
            [ 0.30980393,  0.36078432,  0.39411766],
            [ 0.29607844,  0.36078432,  0.40784314],
            [ 0.29803923,  0.3764706 ,  0.42352942]]])
    """

    B, G, R = cv2.split(RGB)
    R_m, G_m, B_m = tsplit(RGB_m)
    M = as_float_array(M)

    del RGB, RGB_m

    # Updating of the green component.
    R_G = R - G
    B_G = B - G

    FIR = np.ones(3) / 3

    B_G_m = np.where(
        B_m == 1,
        np.where(M == 1, _cnv_h(B_G, FIR), _cnv_v(B_G, FIR)),
        0,
    )
    R_G_m = np.where(
        R_m == 1,
        np.where(M == 1, _cnv_h(R_G, FIR), _cnv_v(R_G, FIR)),
        0,
    )

    del B_G, R_G

    G = np.where(R_m == 1, R - R_G_m, G)
    G = np.where(B_m == 1, B - B_G_m, G)

    # Updating of the red and blue components in the green locations.
    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # Red columns.
    R_c = np.any(R_m == 1, axis=0)[np.newaxis] * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)
    # Blue columns.
    B_c = np.any(B_m == 1, axis=0)[np.newaxis] * np.ones(B.shape)

    R_G = R - G
    B_G = B - G

    k_b = np.array([0.5, 0, 0.5])

    R_G_m = np.where(
        np.logical_and(G_m == 1, B_r == 1),
        _cnv_v(R_G, k_b),
        R_G_m,
    )
    R = np.where(np.logical_and(G_m == 1, B_r == 1), G + R_G_m, R)
    R_G_m = np.where(
        np.logical_and(G_m == 1, B_c == 1),
        _cnv_h(R_G, k_b),
        R_G_m,
    )
    R = np.where(np.logical_and(G_m == 1, B_c == 1), G + R_G_m, R)

    del B_r, R_G_m, B_c, R_G

    B_G_m = np.where(
        np.logical_and(G_m == 1, R_r == 1),
        _cnv_v(B_G, k_b),
        B_G_m,
    )
    B = np.where(np.logical_and(G_m == 1, R_r == 1), G + B_G_m, B)
    B_G_m = np.where(
        np.logical_and(G_m == 1, R_c == 1),
        _cnv_h(B_G, k_b),
        B_G_m,
    )
    B = np.where(np.logical_and(G_m == 1, R_c == 1), G + B_G_m, B)

    del B_G_m, R_r, R_c, G_m, B_G

    # Updating of the red (blue) component in the blue (red) locations.
    R_B = R - B
    R_B_m = np.where(
        B_m == 1,
        np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)),
        0,
    )
    R = np.where(B_m == 1, B + R_B_m, R)

    R_B_m = np.where(
        R_m == 1,
        np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)),
        0,
    )
    B = np.where(R_m == 1, R - R_B_m, B)

    del R_B, R_B_m, R_m

    return cv2.merge([R, G, B])

RGB = cv2.imread("foo_30x18.jpg")
CFA = mosaicing_CFA_Bayer(RGB)
demosaic = demosaicing_CFA_Bayer_Menon2007(CFA, 'RGGB')

# resized = cv2.resize(RGB, (30, 18), interpolation = cv2.INTER_AREA)
# cv2.imwrite("foo_30x18.jpg", resized)

cv2.imshow("origin", RGB)
cv2.imshow("cfa", CFA)
cv2.imshow("demosaic", demosaic)

cv2.waitKey(0)
