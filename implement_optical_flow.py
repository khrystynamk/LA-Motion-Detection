import numpy as np
import sys
import scipy.ndimage
import cv2
import skimage.io
import skimage.transform
from functools import partial
import matplotlib.pyplot as plt
import scipy.fftpack, scipy.fft

# ------------------------------
# Compute the local polynomial expansion of a 2D signal.

#     Parameters:
#     - f (numpy.ndarray) : Input signal (2D array)
#     - c (float) : Certainty of the signal
#     - sigma (float) : Standard deviation of the applicability Gaussian kernel

#     Returns:
#     - A (numpy.ndarray) : Quadratic term of the polynomial expansion (Symmetric matrix)
#     - B (numpy.ndarray) : Linear term of the polynomial expansion (Vector)
#     - C (float) : Constant term of the polynomial expansion
# ------------------------------


def poly_exp(f, c, sigma):
    n = int(4 * sigma + 1)
    x = np.arange(-n, n + 1, dtype=int)
    a = np.exp(-(x**2) / (2 * sigma**2))

    bx = np.stack(
        [np.ones(a.shape), x, np.ones(a.shape), x**2, np.ones(a.shape), x], axis=-1
    )
    by = np.stack(
        [np.ones(a.shape), np.ones(a.shape), x, np.ones(a.shape), x**2, x], axis=-1
    )

    cf = c * f

    G = np.empty(list(f.shape) + [bx.shape[-1]] * 2)
    v = np.empty(list(f.shape) + [bx.shape[-1]])

    abx = a[:, np.newaxis] * bx
    aby = a[:, np.newaxis] * by
    abbx = abx[:, :, None] * bx[:, None, :]
    abby = aby[:, :, None] * by[:, None, :]

    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.convolve1d(c, abbx[..., i, j][::-1], axis=0, cval=0)
            G[..., i, j] = scipy.ndimage.convolve1d(G[..., i, j], abby[..., i, j][::-1], axis=1, cval=0)

        v[..., i] = scipy.ndimage.convolve1d(cf, abx[..., i][::-1], axis=0, cval=0)
        v[..., i] = scipy.ndimage.convolve1d(v[..., i], aby[..., i][::-1], axis=1, cval=0)

    r = np.linalg.solve(G, v)

    A = np.empty(list(f.shape) + [2, 2])
    A[..., 0, 0] = r[..., 3]
    A[..., 0, 1] = r[..., 5] / 2
    A[..., 1, 0] = A[..., 0, 1]
    A[..., 1, 1] = r[..., 4]

    B = np.empty(list(f.shape) + [2])
    B[..., 0] = r[..., 1]
    B[..., 1] = r[..., 2]

    C = r[..., 0]

    return A, B, C


# ------------------------------
# Calculate the Gunnar Farneback optical flow.

#     Parameters:
#     f1 (numpy.ndarray) : First frame
#     f2 (numpy.ndarray) : Second frame
#     sigma (float) : Polynomial expansion applicability Gaussian kernel sigma
#     c1 (numpy.ndarray) : Certainty of first image
#     c2 (numpy.ndarray) : Certainty of second image
#     sigma_flow (float) : Applicability window Gaussian kernel sigma for polynomial matching
#     num_iter (int) : Number of iterations to run (defaults to 1)
#     d (numpy.ndarray) (optional) : Initial displacement field

#     Returns:
#     d (numpy.ndarray) : Optical flow field. d[i, j] is the (y, x) displacement for pixel (i, j)
# ------------------------------

def flow_iterative(f1, f2, sigma, c1, c2, sigma_flow, num_iter=1, d=None):

    A1, B1, _ = poly_exp(f1, c1, sigma)
    A2, B2, _ = poly_exp(f2, c2, sigma)

    # Pixel coordinates of each point in the images
    x = np.stack(
        np.broadcast_arrays(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])),
        axis=-1,
    ).astype(int)

    if d is None:
        d = np.zeros(list(f1.shape) + [2])

    # Applicability convolution window
    n_flow = int(4 * sigma_flow + 1)
    xw = np.arange(-n_flow, n_flow + 1)
    w = np.exp(-(xw**2) / (2 * sigma_flow**2))

    # The model is constant by default
    S = np.eye(2)
    S_T = S.swapaxes(-1, -2)

    for _ in range(num_iter):
        # d~ - displacement field fit to nearest pixel
        d_ = d.astype(int)
        x_ = x + d_

        # Constrain d~ to be on-image
        # Find points that would have been off-image
        x_2 = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)
        off_f = np.any(x_ != x_2, axis=-1)
        x_ = x_2

        # Set certainty to 0 for off-image points
        c_ = c1[x_[..., 0], x_[..., 1]]
        c_[off_f] = 0

        # Calculate A, deltaB for each point, according to paper
        A = (A1 + A2[x_[..., 0], x_[..., 1]]) / 2
        A *= c_[
            ..., None, None
        ]

        delB = -1 / 2 * (B2[x_[..., 0], x_[..., 1]] - B1) + (A @ d_[..., None])[..., 0]
        delB *= c_[
            ..., None
        ]

        A_T = A.swapaxes(-1, -2)
        ATA = S_T @ A_T @ A @ S
        ATb = (S_T @ A_T @ delB[..., None])[..., 0]

        # Apply convolution to calculate linear equation for each pixel: G*d = h
        G = scipy.ndimage.convolve1d(ATA, w, axis=0, cval=0)
        G = scipy.ndimage.convolve1d(G, w, axis=1, cval=0)

        h = scipy.ndimage.convolve1d(ATb, w, axis=0, cval=0)
        h = scipy.ndimage.convolve1d(h, w, axis=1, cval=0)

        d = (S @ np.linalg.solve(G, h)[..., None])[..., 0]

    return d

def preprocess_frames(video_path):
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

    f1 = gray_prev.astype(np.double)
    f2 = gray.astype(np.double)

    return f1, f2

def main(video_path):
    f1, f2 = preprocess_frames(video_path)

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
    pyramid_layers = 1
    options = dict(
        sigma=4.0,
        sigma_flow=4.0,
        num_iter=3
    )
    flow_field = None

    for pyr1, pyr2, c1_, c2_ in reversed(
        list(
            zip(
                *list(
                    map(
                        partial(skimage.transform.pyramid_gaussian, max_layer=pyramid_layers),
                        [f1, f2, c1, c2],
                    )
                )
            )
        )
    ):
        if flow_field is not None:
            flow_field = skimage.transform.pyramid_expand(flow_field, channel_axis=-1)
            flow_field = flow_field[: pyr1.shape[0], : pyr2.shape[1]] * 2
        flow_field = flow_iterative(pyr1, pyr2, c1=c1_, c2=c2_, d=flow_field, **options)

    options_cv = dict(
        pyr_scale=0.5,
        levels=4,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    flow_field_cv = cv2.calcOpticalFlowFarneback(
        f2.astype(np.uint8), f1.astype(np.uint8), None, **options_cv
    )
    flow_field_cv = -flow_field_cv[..., (1, 0)]

    # warped frames
    xw = flow_field + np.moveaxis(np.indices(f1.shape), 0, -1)
    xw2 = flow_field_cv + np.moveaxis(np.indices(f1.shape), 0, -1)
    f2_w2 = skimage.transform.warp(f2, np.moveaxis(xw2, -1, 0), cval=np.nan)
    f2_w = skimage.transform.warp(f2, np.moveaxis(xw, -1, 0), cval=np.nan)

    _, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    p = 2.0
    vmin, vmax = np.nanpercentile(f1 - f2, [p, 100 - p])

    axes[0, 0].imshow(f1, cmap="gray")
    axes[0, 0].set_title("f1 Origin Image)")
    axes[0, 1].imshow(f2, cmap="gray")
    axes[0, 1].set_title("f2 Moving Image")
    axes[1, 0].imshow(f1 - f2_w2, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("f1 - f2 Opencv")
    axes[1, 1].imshow(f1 - f2_w, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("f1 - f2 Our implementation")

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 implement_optical_flow.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])