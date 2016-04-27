You need to build an OpenCV project, with Dlib Library for Facial Landmark Points Detection, available at: https://sourceforge.net/projects/dclib/files/dlib/v18.10/dlib-18.10.tar.bz2/download

We need the DLib Detector to return the Landmark Points as vector of Points. For this, replace the "full_object_detection.h" file of Dlib Library with the one given here.

Now add the "main_for_shots.cpp" file in the OpenCV project with DLib. Add your video and its "shot_info.txt" generated in step-1 at the cpp file's location.

