import cv2
# type hints 
from typing import Tuple, List
import numpy as np 



def scan_frame(frame, classifier) -> Tuple[bool, np.ndarray]: 


    # convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Set min face size to 60 by 60 pixels
    faces = classifier.detectMultiScale(gray_frame, 1.1, 5, minSize = (60, 60))


    # return empty array if no face is found
    if len(faces) == 0: 
        return frame, []
    

    return frame, faces



# input type: video file
def read_vid(video): 


    framecount = 1


    # initialize Haar cascade for face detection
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


    img_lst = [] 


    vid = cv2.VideoCapture(video)


    # collect cropped face frames
    while True: 

        cond, img = vid.read()


        # stop loop if video is under 64 frames
        if not cond: 
            return None


        # face in form of NumPy array and captured frame
        frame, faces = scan_frame(img, classifier)


        f_height = frame.shape[0]
        f_width = frame.shape[1]
        # print(f_width, f_height)


        # Crop face 
        if len(faces) > 0: 
            x, y, w, h = faces[0]

            # create padding around face 
            pad = int(0.1 * w) 
            x_bot = max(x - pad, 0) 
            x_top = min(x + w + pad, f_width)
            y_bot = max(y - pad, 0) 
            y_top = min(y + h + pad, f_height)

            crop_img = frame[y_bot:y_top, x_bot:x_top]
            # img_test = cv2.imwrite(f"C:/Users/Samko/Project/deepfakeproj/img{framecount}.png", crop_img)


            # capture 64 frames from video
            if framecount <= 64: 
                img_lst.append(crop_img) 
                framecount += 1


            else: 
                return img_lst