from utils import download_link, load_pred
from tkinter import Tk
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename
import os


def main(): 
    
    root = Tk() 


    # for url processing (eventually will implement everything to Flask app) 

    input = simpledialog.askstring("Deepface Detection System for Faces", "Is your video a link or file?")


    if input.lower() == "link": 

        url = simpledialog.askstring("Video Link", "Enter the video URL:")


        download_link(url) 


        file_path = os.path.join(os.getcwd(), "video_dfds.mp4") 


        return load_pred(file_path) 


    else: 
        
         # for uploading files 
        file_path = askopenfilename(
            title = "Please select a video file to process", 
            filetypes = [("Videos", "*.mp4")]
        ) 


        if not file_path: 
            print("No file selected. Shutting down program.")
            exit() 


        return load_pred(file_path) 





if __name__ == "__main__": 

    main() 
    