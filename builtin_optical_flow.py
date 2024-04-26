import cv2 as cv
import numpy as np
import sys

def read_first_frame(video_path):
    cap = cv.VideoCapture(video_path)
    _, first_frame = cap.read()
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    return cap, prev_gray

# ------------------------------
# The draw_flow function is designed to visualize the vectors of the optical stream in the image. 
# This is done by drawing lines showing the movement of each pixel between two consecutive video frames.
# ------------------------------

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

# --------------------------------
# Funtion is used to render an optical flow in HSV format for display.
# --------------------------------

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

# --------------------------------
# Funtion is used to draw contours (boxes) around moving objects.
# --------------------------------

def draw_contours(flow, frame):

    magnitude, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)
    thresholded_mag = cv.threshold(magnitude, 80, 255, cv.THRESH_BINARY)[1]

    contours, _ = cv.findContours(thresholded_mag, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    min_contour_area = 500
    large_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]
    frame_out = frame.copy()
    for cnt in large_contours:
        x, y, w, h = cv.boundingRect(cnt)
        frame_out = cv.rectangle(frame, (x, y), (x+w, y+h), (227, 28, 190), 3)
    return frame_out

# --------------------------------
# Funtion is used to calcOpticalFlowFarneback
# --------------------------------

def optical_flow_detection(cap, prev_gray):

    def apply_pca_to_flow(flow_reshaped):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        std_scaler = StandardScaler()
        scaled_flow = std_scaler.fit_transform(flow_reshaped)
        
        pca = PCA(n_components=1)
        pca.fit(scaled_flow)
        print(sum(pca.explained_variance_ratio_))

        reconstructed_flow = pca.inverse_transform(pca.transform(scaled_flow))

        return reconstructed_flow.reshape(flow_reshaped.shape)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # cv.imshow("input", frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        cv.imshow('contours', draw_contours(flow, frame))
        cv.imshow('flow', draw_flow(gray, flow))
        # cv.imshow('flow HSV', draw_hsv(flow))

        flow_reshaped = flow.reshape(-1, 2)
        flow_pca = apply_pca_to_flow(flow_reshaped).reshape(flow.shape)
        cv.imshow('flow pca', draw_flow(gray, flow_pca))
        cv.imshow('contours_pca', draw_contours(flow_pca, frame))

        prev_gray = gray

        # Frames are read by intervals of 1 millisecond
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def release_resources(cap):
    cap.release()
    cv.destroyAllWindows()

def main(video_path):
    cap, prev_gray, mask = read_first_frame(video_path)
    # cap, prev_gray = read_first_frame(0)
    optical_flow_detection(cap, prev_gray)
    release_resources(cap)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 builtin_optical_flow.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])