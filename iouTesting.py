import cv2 as cv

def intersection_over_union(boxA, boxB):
    # Calculates the Intersection over Union (IoU) of two bounding boxes.

    # Parameters:

    # boxA (list of int): The coordinates [x1, y1, x2, y2] of the first bounding box.
    # boxB (list of int): The coordinates [x1, y1, x2, y2] of the second bounding box.

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def get_bounding_boxes(image_path, truthGrounding, predictedGrounding):
    # Processes an image to draw bounding boxes and compute the IoU. Displays the image with the bounding boxes and IoU annotation.
    
    image = cv.imread(image_path)
    if image is None:
        print("Error loading image")
        return

    # Draw the ground-truth bounding box and the predicted bounding box
    cv.rectangle(image, (truthGrounding[0], truthGrounding[1]), (truthGrounding[2], truthGrounding[3]), (225,203, 30), 2)
    cv.rectangle(image, (predictedGrounding[0], predictedGrounding[1]), (predictedGrounding[2], predictedGrounding[3]), (139,55, 200), 2)
    
    # Compute the intersection over union and display it
    iou = intersection_over_union(truthGrounding, predictedGrounding)
    cv.putText(image, f"IoU: {iou:.4f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (225,203, 30), 2)

    # Save the image to a file
    cv.imwrite('output_image.png', image)



if __name__ == "__main__":
    # get_bounding_boxes("./frame/origin_frame1.jpg", [6, 325, 359, 492], [3, 328, 382, 503]) # testing first frame base on Convolution 
    # get_bounding_boxes("./frame/origin_frame2.jpg", [121, 320, 428, 465], [90, 336, 446, 486]) # testing second frame base on Convolution 
    # get_bounding_boxes("./frame/origin_frame1.jpg", [6, 325, 359, 492], [10, 293, 355, 490]) # testing first frame base on FFT
    get_bounding_boxes("./frame/origin_frame2.jpg", [121, 320, 428, 465], [107, 289, 429, 483]) # testing second frame base on FFT 








