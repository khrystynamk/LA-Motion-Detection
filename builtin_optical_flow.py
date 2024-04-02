import cv2 as cv
import numpy as np
import sys

def read_first_frame(video_path):
    # Read video feed
    cap = cv.VideoCapture(video_path)
    ret, first_frame = cap.read()
    # Converts frame to grayscale, less computationally expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255
    return cap, prev_gray, mask

def optical_flow_detection(cap, prev_gray, mask):
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        cv.imshow("input", frame)
        # Convert each frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculate dense optical flow by Gunner-Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # Set image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Set image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        cv.imshow("dense optical flow", rgb)
        prev_gray = gray
        
        # Frames are read by intervals of 1 millisecond
        if cv.waitKey(1) & 0xFF == ord('q'):
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
