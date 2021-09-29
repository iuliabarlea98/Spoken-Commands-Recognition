# Spoken-Commands-Recognition
University project
Purpose: recognition of mini speech commands from a dataset containing 8 types of English spoken commands, each of them having 1000 .WAV audio samples.
-feature extraction is achived by using the MFCC, first derivative MFCC , second derivative MFCC
-recognition using a Convolutional Neural Networks model
-a new command is added, containing a Romanian command, added to the training set.
-the new command is created on 67 original audio samples, own recordings, and then tranformed into 1000 samples as the other commands, using Audacity to change the pitch of the voices
-USE CASE: after the recognition is successfully done, the model is integrated into a GUI, using 4 of the commands (up/down/left/right) for moving a picture into these directions, according to the model's predictions. The movement is initialized using W/A/S/D from the keyboard. 

