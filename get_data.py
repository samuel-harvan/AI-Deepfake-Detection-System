import os
from PIL import Image
from detection import read_vid
import json


# saved data format - data_paths: list[str] and labels: list[float]
def find_frames(limit) -> None: 

    # file paths 
    # using two most common deepfake techniques from possible 6 deepfake folders
    real_vids = "/Users/Samko/Downloads/FaceForensics++_C23/original"
    face2face = "/Users/Samko/Downloads/FaceForensics++_C23/Face2Face"
    deepfakes = "/Users/Samko/Downloads/FaceForensics++_C23/Deepfakes"


    data_paths = [] 
    labels = [] 
    

    # for file naming 
    framecounter = 1


    # iterate through fake videos
    for fakes in [face2face, deepfakes]: 

        for vid in os.listdir(fakes):

            vid_path = os.path.join(fakes, vid)


            # collecting frames from video 
            frames = read_vid(vid_path, limit)


            if frames is not None: 

                for frame in frames:

                    frame_path = f"data/fake{framecounter}.jpg"


                    # save frame as jpeg file to data folder 
                    Image.fromarray(frame).save(frame_path) 


                    if framecounter % 2000 == 0: 
                        print(f"{framecounter} frames collected.")


                    data_paths.append(frame_path)
                    labels.append(1.0) # 1.0 for fake labelling 
                    framecounter += 1

            
            else: 

                continue   


    print("Done fake, beginning real.")


    # iterate through real videos          
    for real in os.listdir(real_vids): 

        vid_path =  os.path.join(real_vids, real)


        # collecting frames from video
        frames = read_vid(vid_path, limit)


        if frames is not None: 

            for frame in frames:

                frame_path = f"data/real{framecounter}.jpg"


                # save frame as jpeg file to data folder
                Image.fromarray(frame).save(frame_path) 


                if framecounter % 2000 == 0:
                    print(f"{framecounter} frames collected.")


                data_paths.append(frame_path)
                labels.append(0.0) # 0.0 for real labelling 
                framecounter += 1


        else: 

            continue


    # store variables locally to be accessed later for training 
    with open("stored_vars/xy_var.json", "w") as var_file:

        json.dump({"paths": data_paths, "labels": labels}, var_file) 


#find_frames(10) 