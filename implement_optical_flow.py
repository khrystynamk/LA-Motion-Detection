import numpy as np
import sys
import scipy.ndimage
import cv2
import skimage.io
import skimage.transform
from functools import partial
import matplotlib.pyplot as plt
from builtin_optical_flow import read_first_frame, release_resources


def poly_exp(f, c, sigma):
    """
    Calculates the local polynomial expansion of a 2D signal, as described by Farneback

    Uses separable normalized correlation

    $f ~ x^T A x + B^T x + C$

    If f[i, j] and c[i, j] are the signal value and certainty of pixel (i, j) then
    A[i, j] is a 2x2 array representing the quadratic term of the polynomial, B[i, j]
    is a 2-element array representing the linear term, and C[i, j] is a scalar
    representing the constant term.

    Parameters
    ----------
    f
        Input signal
    c
        Certainty of signal
    sigma
        Standard deviation of applicability Gaussian kernel

    Returns
    -------
    A
        Quadratic term of polynomial expansion
    B
        Linear term of polynomial expansion
    C
        Constant term of polynomial expansion
    """

    # ----- [Equivalent Correlation Kernels section in the paper] -----

    # Calculate applicability kernel (1D because it is separable, computation is significantly more efficient)
    n = int(4 * sigma + 1)
    x = np.arange(-n, n + 1, dtype=int)
    a = np.exp(-(x**2) / (2 * sigma**2))  # a: applicability kernel [n]

    # b: calculate b from the paper. Calculate separately for X and Y dimensions
    # [n, 6]
    # bx array has a shape determined by the shape of the applicability kernel a 
    # with an additional dimension for the different terms in the polynomial basis

    # ----- [Estimating the Coefficients of a Polynomial Model] -----
    # polynomial basis, {1, x, y, x^2, y^2, xy}
    # y = 1
    bx = np.stack(
        [np.ones(a.shape), x, np.ones(a.shape), x**2, np.ones(a.shape), x], axis=-1
    )
    by = np.stack(
        [
            np.ones(a.shape),
            np.ones(a.shape),
            x,
            np.ones(a.shape),
            x**2,
            x,
        ],
        axis=-1,
    )

    # Pre-calculate product of certainty and signal
    cf = c * f

    # ----- [Cartesian Separability section in the paper] -----
    # The goal is to find the coefficients of a second-order polynomial that best fits the local signal.

    # G and v are used to calculate "r" from the paper: r = G^(-1)*v -> v = G*r
    # r is the parametrization of the 2nd order polynomial for f
    G = np.empty(list(f.shape) + [bx.shape[-1]] * 2)
    v = np.empty(list(f.shape) + [bx.shape[-1]])

    # Apply separable cross-correlations

    # Pre-calculate quantities recommended in paper
    # einsum does multiplication, summation and transposition faster
    # i,ij->ij A has one axis i, B has 2 axes (i and j) -- dimension labels
    ab = np.einsum("i,ij->ij", a, bx) # inner product (a · bm), this can be rewritten as (a[:, np.newaxis] * bx)
    abb = np.einsum("ij,ik->ijk", ab, bx) # inner product (a · bm, bm), this can be rewritten as abb = np.matmul(ab[:, :, None], bx[:, None, :]) or ab[:, :, None] @ bx[:, None, :], where @ is shortcut for matmul

    # Calculate G and v for each pixel x with cross-correlation
    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(
                c, abb[..., i, j], axis=0, mode="constant", cval=0 #‘constant’ (k k k k | a b c d | k k k k)
                # The input is extended by filling all values beyond the edge with cval
            )

        v[..., i] = scipy.ndimage.correlate1d(
            cf, ab[..., i], axis=0, mode="constant", cval=0
        )

    # Pre-calculate quantities recommended in paper
    ab = np.einsum("i,ij->ij", a, by) # inner product (a · bk), can be rewritten as (a[:, np.newaxis] * by)
    abb = np.einsum("ij,ik->ijk", ab, by) # inner product (a · bk, bk), can be rewritten as abb = np.matmul(ab[:, :, None], by[:, None, :]) or ab[:, :, None] @ by[:, None, :], where @ is shortcut for matmul

    # Calculate G and v for each pixel y with cross-correlation
    # [..., i, j] is a shortcut to indicate that all axes preceding or following it should be fully included

    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(
                G[..., i, j], abb[..., i, j], axis=1, mode="constant", cval=0
            )

        v[..., i] = scipy.ndimage.correlate1d(
            v[..., i], ab[..., i], axis=1, mode="constant", cval=0
        )

    # Solve r for each pixel
    r = np.linalg.solve(G, v)

    # ----- [Estimating the Coefficients of a Polynomial Model section in the paper] -----

    # Quadratic term
    A = np.empty(list(f.shape) + [2, 2])
    A[..., 0, 0] = r[..., 3] # r_4
    A[..., 0, 1] = r[..., 5] / 2 # r_6 / 2
    A[..., 1, 0] = A[..., 0, 1] # r_6 / 2
    A[..., 1, 1] = r[..., 4] # r_5

    # Linear term
    B = np.empty(list(f.shape) + [2])
    B[..., 0] = r[..., 1] # r_2
    B[..., 1] = r[..., 2] # r_3

    # constant term
    C = r[..., 0] # r_1

    return A, B, C


def flow_iterative(
    f1, f2, sigma, c1, c2, sigma_flow, num_iter=1, d=None
):
    """
    Calculates optical flow with an algorithm described by Gunnar Farneback

    Parameters
    ----------
    f1
        First image
    f2
        Second image
    sigma
        Polynomial expansion applicability Gaussian kernel sigma
    c1
        Certainty of first image
    c2
        Certainty of second image
    sigma_flow
        Applicability window Gaussian kernel sigma for polynomial matching
    num_iter
        Number of iterations to run (defaults to 1)
    d: (optional)
        Initial displacement field

    Returns
    -------
    d
        Optical flow field. d[i, j] is the (y, x) displacement for pixel (i, j)
    """

    # Calculate the polynomial expansion at each point in the images
    A1, B1, C1 = poly_exp(f1, c1, sigma)
    A2, B2, C2 = poly_exp(f2, c2, sigma)

    # Pixel coordinates of each point in the images
    x = np.stack(
        np.broadcast_arrays(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])),
        axis=-1,
    ).astype(int)

    # Initialize displacement field
    if d is None:
        d = np.zeros(list(f1.shape) + [2])

    # Set up applicability convolution window
    n_flow = int(4 * sigma_flow + 1)
    xw = np.arange(-n_flow, n_flow + 1)
    w = np.exp(-(xw**2) / (2 * sigma_flow**2))

    # The model is constant by default
    S = np.eye(2)
    S_T = S.swapaxes(-1, -2)

    # Iterate convolutions to estimate the optical flow
    for _ in range(num_iter):
        # Set d~ as displacement field fit to nearest pixel (and constrain to not
        # being off image). Note we are setting certainty to 0 for points that
        # would have been off-image had we not constrained them
        d_ = d.astype(int)
        x_ = x + d_

        # x_ = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)

        # Constrain d~ to be on-image, and find points that would have
        # been off-image
        x_2 = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)
        off_f = np.any(x_ != x_2, axis=-1)
        x_ = x_2

        # Set certainty to 0 for off-image points
        c_ = c1[x_[..., 0], x_[..., 1]]
        c_[off_f] = 0

        # Calculate A and delB for each point, according to paper, add in certainty by applying to A and delB
        A = (A1 + A2[x_[..., 0], x_[..., 1]]) / 2
        A *= c_[
            ..., None, None
        ]

        delB = -1 / 2 * (B2[x_[..., 0], x_[..., 1]] - B1) + (A @ d_[..., None])[..., 0]
        delB *= c_[
            ..., None
        ]

        # Pre-calculate quantities recommended by paper
        A_T = A.swapaxes(-1, -2)
        ATA = S_T @ A_T @ A @ S
        ATb = (S_T @ A_T @ delB[..., None])[..., 0]
        # btb = delB.swapaxes(-1, -2) @ delB

        # Apply separable cross-correlation to calculate linear equation
        # for each pixel: G*d = h
        G = scipy.ndimage.correlate1d(ATA, w, axis=0, mode="constant", cval=0)
        G = scipy.ndimage.correlate1d(G, w, axis=1, mode="constant", cval=0)

        h = scipy.ndimage.correlate1d(ATb, w, axis=0, mode="constant", cval=0)
        h = scipy.ndimage.correlate1d(h, w, axis=1, mode="constant", cval=0)

        d = (S @ np.linalg.solve(G, h)[..., None])[..., 0]

    return d

def main(video_path):
    """
    Compares this implementation of Farneback's algorithms to OpenCV's implementation
    of a similar version of the algorithm
    """

    # ---------------------------------------------------------------
    # get images to calculate flow for
    # ---------------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    if not ret:
        print("Error reading video frame")
        return

    ret, frame2 = cap.read()
    if not ret:
        print("Error reading video frame")
        return
    
    gray_prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Convert frames to double precision for optical flow calculation
    f1 = gray_prev.astype(np.double)
    f2 = gray.astype(np.double)

    # certainties for images - certainty is decreased for pixels near the edge
    # of the image, as recommended by Farneback

    # c1 = np.ones_like(f1)
    # c2 = np.ones_like(f2)

    c1 = np.minimum(
        1, 1 / 5 * np.minimum(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1]))
    )
    c1 = np.minimum(
        c1,
        1
        / 5
        * np.minimum(
            f1.shape[0] - 1 - np.arange(f1.shape[0])[:, None],
            f1.shape[1] - 1 - np.arange(f1.shape[1]),
        ),
    )
    c2 = c1

    # ---------------------------------------------------------------
    # calculate optical flow with this algorithm
    # ---------------------------------------------------------------

    n_pyr = 4

    # version using no regularization model
    opts = dict(
        sigma=4.0,
        sigma_flow=4.0,
        num_iter=3
    )

    # optical flow field
    d = None

    # calculate optical flow using pyramids
    # note: reversed(...) because we start with the smallest pyramid
    for pyr1, pyr2, c1_, c2_ in reversed(
        list(
            zip(
                *list(
                    map(
                        partial(skimage.transform.pyramid_gaussian, max_layer=n_pyr),
                        [f1, f2, c1, c2],
                    )
                )
            )
        )
    ):
        if d is not None:
            d = skimage.transform.pyramid_expand(d, channel_axis=-1)
            d = d[: pyr1.shape[0], : pyr2.shape[1]] * 2

        d = flow_iterative(pyr1, pyr2, c1=c1_, c2=c2_, d=d, **opts)

    xw = d + np.moveaxis(np.indices(f1.shape), 0, -1)

    # ---------------------------------------------------------------
    # calculate optical flow with opencv
    # ---------------------------------------------------------------

    opts_cv = dict(
        pyr_scale=0.5,
        levels=6,
        winsize=25,
        iterations=10,
        poly_n=25,
        poly_sigma=3.0,
        # flags=0
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )

    d2 = cv2.calcOpticalFlowFarneback(
        f2.astype(np.uint8), f1.astype(np.uint8), None, **opts_cv
    )
    d2 = -d2[..., (1, 0)]

    xw2 = d2 + np.moveaxis(np.indices(f1.shape), 0, -1)

    # ---------------------------------------------------------------
    # use calculated optical flow to warp images
    # ---------------------------------------------------------------

    # opencv warped frame
    f2_w2 = skimage.transform.warp(f2, np.moveaxis(xw2, -1, 0), cval=np.nan)

    # warped frame
    f2_w = skimage.transform.warp(f2, np.moveaxis(xw, -1, 0), cval=np.nan)

    # ---------------------------------------------------------------
    # visualize results
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    p = 2.0  # percentile of histogram edges to chop off
    vmin, vmax = np.nanpercentile(f1 - f2, [p, 100 - p])
    cmap = "gray"

    axes[0, 0].imshow(f1, cmap=cmap)
    axes[0, 0].set_title("f1 (fixed image)")
    axes[0, 1].imshow(f2, cmap=cmap)
    axes[0, 1].set_title("f2 (moving image)")
    axes[1, 0].imshow(f1 - f2_w2, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("difference f1 - f2 warped: opencv implementation")
    axes[1, 1].imshow(f1 - f2_w, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("difference f1 - f2 warped: this implementation")

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 implement_optical_flow.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])