import librosa
import tensorflow as tf
import numpy as np
import speech_recognition as sr
import pyaudio
import pyttsx3
from googletrans import Translator, constants
from playsound import playsound
SAVED_MODEL_PATH = "modelrecognition.h5"
SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """

    model = None
    _mapping = [
        "down",
        "go",
        "left",
        "no",
        "right",
        "stai",
        "stop",
        "up",
        "yes",
        
    ]
    _instance = None


    def predict(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract MFCC
        allMFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        allMFCCs = allMFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(allMFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
              #extract first derivative of MFCCs
            MFCCs_Delta = librosa.feature.delta(MFCCs, order=1)
            #extract second derivative of MFCCs
            MFCC_Delta2 = librosa.feature.delta(MFCCs, order=2)
                    #concatenation
            allMFCCs=np.concatenate((MFCCs,MFCCs_Delta,MFCC_Delta2))
        return allMFCCs.T


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance




if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()
    assert kss is kss1

     #init the Google API translator
    translator = Translator()
    engine=pyttsx3.init()
    yt=sr.AudioFile('D:/proiect PSV/testare/bgbggbgbgbg.wav')
    r = sr.Recognizer()
    keyWord = 'stai'
    with yt as source:
       print('I am listening to the audio file\n')
       playsound('D:/proiect PSV/testare/bgbggbgbgbg.wav')
       audio = r.record(source)
       text = r.recognize_google(audio,language='ro-RO') 
       engine.say(text=keyWord)
       translation = translator.translate(keyWord, src="ro",dest="en")
       print('This word is not in english , it means : ')
       print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
       if keyWord.lower() in text.lower():
          print('Keyword detected in the speech.')
          engine.runAndWait()
          keyword = kss.predict("D:/proiect PSV/testare/bgbggbgbgbg.wav")
          print('The predicted keyword is :  '+ keyword)
                     
    yt1=sr.AudioFile("D:/proiect PSV/testare/yes.wav")
    keyWord1='yes'
    with yt1 as source:
        print('I am listening to another audio file')
        playsound('D:/proiect PSV/testare/yes.wav')
        audio1=r.record(source)
        text1=r.recognize_google(audio1,language='en')
        if keyWord1.lower() in text1.lower():
         print("Keyword detected in the speech!")
         keyword1 = kss1.predict("D:/proiect PSV/testare/yes.wav")
         print('I want to predict another word : ')
         print('It is: '+keyword1)
        
    print(' ')
    print('Now, let us use these commands ! ')
    print('As directions: ')
    print('Press W/A/S/D as the window appears: ')
    up=Keyword_Spotting_Service()
    down=Keyword_Spotting_Service()
    left=Keyword_Spotting_Service()
    right=Keyword_Spotting_Service()
    
    
    from tkinter import *

    def move_left(event):
        keyword2 = left.predict("D:/proiect PSV/testare/left1.wav")
        print('This will allow the stickman to go  : '+ keyword2)
        label.place(x=label.winfo_x()-10,y=label.winfo_y())
    def move_down(event):
        keyword3 = down.predict("D:/proiect PSV/testare/down.wav")
        print('This will allow the stickman to go  : '+ keyword3)
        label.place(x=label.winfo_x(),y=label.winfo_y()+10)
    def move_up(event):
        keyword4 = up.predict("D:/proiect PSV/testare/up.wav")
        print('This will allow the stickman to go  : '+ keyword4)
        label.place(x=label.winfo_x(),y=label.winfo_y()-10)
    def move_right(event):
        keyword5 = right.predict("D:/proiect PSV/testare/right1.wav")
        print('This will allow the stickman to go  : '+ keyword5)
        label.place(x=label.winfo_x()+10,y=label.winfo_y())
    window=Tk()
    window.geometry("600x600")
    window.bind("<w>",move_up)
    window.bind("<s>",move_down)
    window.bind("<a>",move_left)
    window.bind("<d>",move_right)
    myimage=PhotoImage(file='C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/proiect PSV/stickkk.png')
    label=Label(window,image=myimage)
    label.place(x=10,y=10)
    window.mainloop()
    
    
                 
                            