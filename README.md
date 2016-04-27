# Face-Detection-Tracking-and-Clustering
We detect and track faces in video, then extract features from those face tracks 
and try to cluster them into given number of Clusters, each Cluster representing a unique person.
To get a full idea of our project, refer to the file "ppt_explaining_the_project.pdf". Results were obtained on video:
https://youtu.be/A1fVcj29xhk

Running the project on a video-file involves 4 steps:

1) Shot Detection:

We need the information of shots from the video. The shot-detection is done using Python.

Application used is ShotDetect by John Mathe: https://github.com/johmathe/Shotdetect
ReadMe on this GitHub page describes how to use the app via command line.

This app is available for Ubuntu: sudo apt-get install ShotDetect

After running app on your video, a folder will be generated. Paste the 
"xml_parser.py" file (available here) in that folder and run it. 

A file by name "shot_info.txt" will be generated. This file contains shot information
in a way understandable by our Face Tracking code. Copy this file and paste it
at the location of "main_for_shots.cpp" 


2) Detection and Tracking:

You need to build an OpenCV project, 
with Dlib Library for Facial Landmark Points Detection, available at:
https://sourceforge.net/projects/dclib/files/dlib/v18.10/dlib-18.10.tar.bz2/download

We need the DLib Detector to return the Landmark Points as vector of Points. 
For this, replace the "full_object_detection.h" file of Dlib Library with the one
given here.

Now add the "main_for_shots.cpp" file in the OpenCV project with DLib. Add your video and its "shot_info.txt" generated in step-1 
at the cpp file's location. 

3) Feature Extraction:

Copy the "data" folder created in Step-2, and 
paste it at the location of the Feature Extraction cpp files. 
Run the .cpp files then.

4) Clustering:

Is done in Python. Copy the folders generated in step3 and paste them where the Python files are.

If features are generated using "avg_img_features.cpp", run "Clustering_basic_feats.py"
results will be stored in "resultsAVG/".

If features are generated using "3D_track_features.cpp", run "Clustering_adv_feats.py"
results will be stored in "results/".
