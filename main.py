from detection_vid import read_vid
# from detection_img import read_img
from process_imgs import img_to_clip 
from predictor import predict
from dlp import download_link 
from tkinter import Tk
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename
from statistics import mean
import os


def main(): 
    
    root = Tk() 



    # for url processing (eventually will implement everything to Flask app) 

    # input = simpledialog.askstring("Deepface Detection System for Faces", "Is your video a link or file?")

    # if choice.lower() == "link": 

    #     url = simpledialog.askstring("Video Link", "Enter the video URL:")

    # download_link(url) 

    # current_working_dir = os.getcwd()

    # file_path = os.path.join(current_working_dir, "video_dfds.mp4") 



    # for uploading files 
    file_path = askopenfilename(
        title = "Please select a video file to process", 
        filetypes = [("Videos", "*.mp4")]
    ) 


    if not file_path: 
        print("No file selected. Shutting down program.")
        exit() 


    results = [] 


    # finds frames with face
    face_frames = read_vid(file_path)


    # if faces cannot be found, end the program
    if len(face_frames) == 0: 
        print("Faces not found in video. Prediction: 'None'")
        exit() 


    # generate clips from frames
    clips = img_to_clip(face_frames) 


    # for now only unpack half of the batches (will change later)
    for batch_number in range(len(clips), 2): 


        # prediction for each batch
        initial_pred = predict(clips, batch_number)

        results.append(initial_pred) 


    #final prediction 
    final_pred = mean(results) 


    #display results 
    if final_pred < 0.6: 
        return "There video is most likely a deepfake."
    else: 
        return "This video is most likely real."



if __name__ == "__main__": 
    main() 
    