from utils import download_link, load_pred
import os
import FreeSimpleGUI as sg  # unnecessary once Flask implementation is complete (used for testing)


# this version will not work on mac.os (next fix)


def main(): 

    while True: 

        print("Please enter if your data in a url or file format. Type 'file' or 'url':")
        option = input() 


        if option == "url": 

            # for uploading links
            url = sg.popup_get_text("Enter video URL: ")


            download_link(url) 

            
            # later fix (video_dfds.mp4 is a test/placeholder)
            file_path = os.path.join(os.getcwd(), "video_dfds.mp4") 


            return print(f"\n{load_pred(file_path)}") 


        elif option == "file": 
            
            # for uploading files 
            file_path = sg.popup_get_file("Select a video file", file_types=(("MP4 files", "*.mp4"),))


            if file_path is None: 
                print("No file selected. Shutting down program.")
                exit() 


            return print(f"\n{load_pred(file_path)}") 
        
        
        elif option == "EXIT": 

            return print("Have a good day :)")
        
        
        else: 

            print("Invalid input registered. Please try again or type 'EXIT' to shut down this program")