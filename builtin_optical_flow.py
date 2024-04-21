import cv2 as cv
import numpy as np
import sys
import time
import pandas as pd
import pca_implementation

def read_first_frame(video_path):
    # Read video feed
    cap = cv.VideoCapture(video_path)
    _, first_frame = cap.read()
    # Converts frame to grayscale, less computationally expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255
    return cap, prev_gray, mask

def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    

    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

def draw_flow_pca(img, flow, mask_pca, threshold, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    colored_lines = lines.copy()
    colored_lines[..., 1] = np.where(mask_pca.mean(axis=1) > threshold, lines[..., 1], 0)
    

    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # cv.polylines(img_bgr, lines, 0, (0, 0, 255))
    cv.polylines(img_bgr, colored_lines, 0, (0, 255, 255))

    for (x1, y1), (_x2, _y2) in colored_lines:
        cv.circle(img_bgr, (x1, y1), 1, (0, 255, 255), -1)

    return img_bgr

def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx+fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang * (180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v * 4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return bgr

def optical_flow_detection(cap, prev_gray, mask):
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        cv.imshow("input", frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # print(mask[..., 0].shape)

        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        cv.imshow('flow', draw_flow(gray, flow))
        cv.imshow('flow HSV', draw_hsv(flow))

        flow_reshaped = flow.reshape(-1, 2)

        mean = np.empty((0))
        mean, eigenvectors, eigenvalues= cv.PCACompute2(flow_reshaped, mean=mean)

        print(eigenvalues) 
        print(eigenvalues.shape) # (2, 1)

        explained_variance_cutoff = 0.9 
        
        try:
            threshold_eigenvalue = np.sum(eigenvalues[:np.where(np.cumsum(eigenvalues) > explained_variance_cutoff)[0][0]])

            print(flow_reshaped.shape) # (45388, 2)


            # break
            mask_pca = np.abs(flow_reshaped.T - eigenvalues @ (eigenvalues.T @ flow_reshaped.T)) > threshold_eigenvalue
            cv.imshow('flow pca', draw_flow_pca(gray, flow, mask_pca, threshold_eigenvalue))

            prev_gray = gray
        
        except IndexError:
            continue
        

        # Frames are read by intervals of 1 millisecond
        key = cv.waitKey(1)
        if key == ord('q'):
            break

def release_resources(cap):
    cap.release()
    cv.destroyAllWindows()

def main(video_path):
    cap, prev_gray, mask = read_first_frame(video_path)
    optical_flow_detection(cap, prev_gray, mask)

    release_resources(cap)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 builtin_optical_flow.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])
