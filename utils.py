import cv2
import numpy as np
import torch
import yt_dlp
from predictor import predict


def load_pred(file_path) -> str: 

     # finds frames with face
    final_pred = predict(file_path)


    # if faces cannot be found, end the program
    if final_pred is None: 

        return "No face frames detected in video. Prediction: 'None'"

    
    else: 

        return f"The video is {final_pred}% fake."





# for video link downloads
def download_link(url) -> None: 

    settings = {
        "format": "best",
        "outtmpl": "video_dfds.mp4",
        "quiet": True
    }


    yt_dlp.YoutubeDL(settings).download(url)