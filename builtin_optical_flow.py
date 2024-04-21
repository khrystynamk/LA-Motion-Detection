import cv2 as cv
import numpy as np
import sys
import pca_implementation  # Assuming pca_implementation.py defines PCA functions

def read_first_frame(video_path):
    cap = cv.VideoCapture(video_path)
    _, first_frame = cap.read()

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255

    return cap, prev_gray, mask

# ------------------------------
# The draw_flow function is designed to visualize the vectors of the optical stream in the image.
# This is done by drawing lines showing the movement of each pixel between two consecutive video frames.
# ------------------------------
def draw_flow(img, flow, step=25):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T 

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5) 

    img_vector = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(img_vector, lines, 0, (0, 255, 255))

    return img_vector


# --------------------------------
# Funtion is used to render an optical flow in HSV format for display.
# --------------------------------
def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]  

    angle = np.arctan2(fy, fx) + np.pi  
    magnitude = np.sqrt(fx * fx + fy * fy)  

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = angle * (180/np.pi/2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(magnitude * 4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return bgr



def draw_flow_pca(img, flow, step=16, threshold=6.0):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T
    magnitude = np.sqrt(fx*fx + fy*fy)

    # Apply magnitude threshold to remove noise
    motion_mask = magnitude > threshold

    lines = np.vstack([x[motion_mask], y[motion_mask], x[motion_mask]-fx[motion_mask], y[motion_mask]-fy[motion_mask]]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(img_bgr, lines, 0, (0, 255, 255))

    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(img_bgr, (x1, y1), 1, (0, 255, 255), -1) 

    return img_bgr


# --------------------------------
# Funtion is used to calcOpticalFlowFarneback
# --------------------------------
def optical_flow_detection(cap, prev_gray, mask):

    def apply_pca_to_flow(flow_reshaped):
        mean, eigenvectors, eigenvalues = cv.PCACompute2(flow_reshaped, mean=None, retainedVariance=0.9)

        print(flow_reshaped)

        score = np.dot(flow_reshaped - mean, eigenvectors)  # Project flow data onto principal components
        reconstructed = np.dot(score, eigenvectors.T) + mean  # Reconstruct flow data using principal components
        return reconstructed.reshape(flow.shape)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow("input", frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])


        cv.imshow('flow', draw_flow(gray, flow))
        cv.imshow('flow HSV', draw_hsv(flow))

        flow_reshaped = flow.reshape(-1, 2)

        # Calculate PCA
        flow_pca = apply_pca_to_flow(flow_reshaped).reshape(flow.shape)


        cv.imshow('flow pca', draw_flow_pca(gray, flow_pca))
        # cv.imshow('flow pca HSV', draw_hsv(flow_pca))

        
        # Frames are read by intervals of 1 millisecond
        key = cv.waitKey(1)
        if key == ord('q'):
            break

def release_resources(cap):
    cap.release()
    cv.destroyAllWindows()

def main(video_path):
    cap, prev_gray, mask = read_first_frame(0)
    optical_flow_detection(cap, prev_gray, mask)

    release_resources(cap)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 builtin_optical_flow.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])
