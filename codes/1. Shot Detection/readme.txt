The shot-detection is done using Python.

Application used is ShotDetect by John Mathe: https://github.com/johmathe/Shotdetect
ReadMe on this GitHub page describes how to use the app via command line.

This app is available for Ubuntu: sudo apt-get install ShotDetect

After running app on your video, a folder will be generated. Paste the 
"xml_parser.py" file (available here) in that folder and run it. 

A file by name "shot_info.txt" will be generated. This file contains shot information
in a way understandable by our Face Tracking code. Copy this file and paste it
at the location of "main_for_shots.cpp" 