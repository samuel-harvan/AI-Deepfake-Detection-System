import cv2
import numpy as np 


def scan_frame(frame, classifier) -> tuple[bool, np.ndarray]: 

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Set min face size to 30 by 30 pixels
    faces = classifier.detectMultiScale(gray_frame, 1.1, 5, minSize = (30, 30))


    # return empty array if no face is found
    if len(faces) == 0: 
        return frame, []
    

    return frame, faces





# input type: video file
def read_vid(video, limit) -> list[np.ndarray]:

    framecount = 1


    # initialize Haar cascade for face detection
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


    img_lst = [] 


    vid = cv2.VideoCapture(video)


    # collect cropped face frames
    while True: 

        cond, img = vid.read()


        # stop loop if video is under targetted amount of frames
        # will not trigger during testing: for user purposes only 
        if not cond: 
            return None


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


            # capture limited amount of frames from video
            if framecount <= limit: 

                # save in RBG format (prevent color distortion later) 
                img_RBG = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                img_lst.append(img_RBG) 
                framecount += 1


            else: 

                return img_lst


        else: 

            continue 