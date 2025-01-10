from __future__ import print_function
import numpy as np
import cv2
from cap_from_youtube import cap_from_youtube


def get_momentum_centroid(contour):
    """
    Calculate the centroid of a contour using image moments.

    Parameters
    ----------
    contour : array_like
        The contour for which to calculate the centroid.

    Returns
    -------
    tuple
        A tuple (cx, cy) representing the x and y coordinates of the centroid.
        If the contour area is zero, returns (0, 0).
    """

    moments = cv2.moments(contour)
    # Avoid division by zero
    if moments["m00"] != 0:  
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0
    return (cx, cy)

def process_frame(
        frame, 
        fgbg, 
        gb_kernel_size = 7, 
        kernel_size = 5, 
        image_threshold = 170
    ):
    """
    Process a frame to extract the foreground, reduce noise and convert to a binary image.

    Parameters
    ----------
    frame : array_like
        The frame to process.
    fgbg : BackgroundSubtractor
        The background subtractor to use.
    gb_kernel_size : int, optional
        The size of the Gaussian blur kernel (default is 7).
    kernel_size : int, optional
        The size of the morphological transformation kernel (default is 5).
    image_threshold : int, optional
        The threshold to use when converting the image to binary (default is 170).

    Returns
    -------
    binary_image : array_like
        The binary image.
    """

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Reduce noise using Gaussian blur and morphological transformations
    blur = cv2.GaussianBlur(fgmask, (gb_kernel_size, gb_kernel_size), 0)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Convert to binary image
    _, binary_image = cv2.threshold(opening, image_threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def detect_and_track_cars(
        rgb_img, 
        binary_image, 
        cars, 
        history, 
        contourn_min=150, 
        centroid_dist_threshold=50
    ):
    """
    Detects and tracks cars in a binary image.

    Parameters
    ----------
    rgb_image : array_like
        The original RGB image.
    binary_image : array_like
        The binary image to process.
    cars : dict
        A dictionary of car objects, where each key is a car id and each value
        is a dictionary containing the car's frames and centroids.
    history : list
        A list of tuples containing the car id and centroid of each car in the
        previous frame.
    contourn_min : int, optional
        The minimum area of a contour to be considered a car (default is 300).

    Returns
    -------
    current_centroids : list
        A list of tuples containing the car id and centroid of each car in the
        current frame.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    current_centroids = []

    for contour in contours:
        if cv2.contourArea(contour) < contourn_min:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        bbox = (x, y, w, h)
        
        cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        car_id = None
        # centroid = (x + w // 2, y + h // 2)
        centroid = get_momentum_centroid(contour)

        closest_centroid_coord = (None, None)
        closest_centroid_dist = np.inf

        # Match current centroid with historical centroids
        for history_list in history:
            for prev_car_id, prev_centroid, _ in history_list:
                dist = np.linalg.norm(np.array(prev_centroid) - np.array(centroid))

                if dist < closest_centroid_dist:
                    closest_centroid_coord = prev_centroid
                    closest_centroid_dist = dist
        
        if closest_centroid_dist < centroid_dist_threshold:
            car_id = prev_car_id
                
            cars[car_id].append(centroid)
                
            # multiply by 30 to convert for cm/s, 
            # 360 to convert for cm/hr, 
            # and convert for km/h
            cm_to_km_per_hour = 0.036
            pixel_length = 7.2
            frames_per_second = 30

            car_vel = round(closest_centroid_dist * pixel_length * frames_per_second * cm_to_km_per_hour)
            cv2.putText(rgb_img, str(car_vel) + 'km/h', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.line(rgb_img, closest_centroid_coord, centroid, (0, 255, 0), 3)
                
        if car_id is None:
            car_id = len(cars)
            cars[car_id] = [centroid]

        current_centroids.append((car_id, centroid, bbox))

    return current_centroids

def count_cars(rgb_img, cars_history, pt1, pt2):
    car_counting = 0

    for car_id, car_centroid, car_bbox in cars_history:
        x, y, w, h = car_bbox
        if y < pt1[1] < y + h and pt1[0] < car_centroid[0] < pt2[0]:
            car_counting += 1

    cv2.line(rgb_img, pt1, pt2, (0, 255, 0), 3)
    cv2.putText(rgb_img, str(car_counting) + ' cars', (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def main():
    # Video initialization
    video_url = 'https://www.youtube.com/watch?v=nt3D26lrkho&ab_channel=VK'
    
    capture = cap_from_youtube(video_url, '720p')
    if not capture.isOpened():
        print('Unable to open video.')
        exit(0)

    # Background subtractor
    fgbg = cv2.createBackgroundSubtractorKNN()

    cars = {}
    history = []

    while True:
        _, frame = capture.read()
        
        if frame is None:
            break

        height, width, _ = frame.shape

        binary_image = process_frame(frame, fgbg)
        
        new_history = detect_and_track_cars(frame, binary_image, cars, history)

        history.append(new_history)

        count_cars(
            frame, 
            history[-1], 
            (0, height//4 + height//2), 
            (width//2 - 50, height//4 + height//2)
        )
        count_cars(
            frame, 
            history[-1], 
            (width//2 + 50, height//4 + height//2), 
            (width, height//4 + height//2)
        )
        
        if len(history) > 3:
            history.pop(0)

        cv2.imshow("Filtered Road", binary_image)
        cv2.imshow("Road", frame)

        if cv2.waitKey(30) in ['q', 27]:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()