#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 21:23:05 2021

@author: josesolla
"""

import os
import pyaudio
import wave
#import time
import ctypes
from gtts import gTTS
from pygame import mixer
  



ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_char_p)

def AhoTTS(object):
    
    def __init__(self):
        
        self.c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        
        try:
            asound = ctypes.cdll.loadLibrary('libasound.so.2')
            asound.snd_lib_error_set_handler(self.c_error_handler)
        except OSError:
            pass
        
        self.chunk = 1024
        #os.chdir('ahotts-code/bin')
        
        samplewidth = 40
        channels = 1
        rate = 44100
        
        self.pa = pyaudio.PyAudio()
        self.audiostream = self.pa.open(format = self.pa.get_format_from_width(samplewidth),
                        channels = channels,
                        rate = rate,
                        output = True)
        
        
    
    def tts(self, text):
        
        ## Write text in txt file
        f=open("tts/ahotts-code/bin/temp/input.txt","w+",encoding ="iso-8859-15") #cp1252 o iso-8859-1
        f.write(text) #Escribimos la entrada en consola en el fichero
        f.close()
        
        # Execute AhoTTS
        os.system('./tts/ahotts-code/bin/tts -InputFile=tts/ahotts-code/bin/temp/input.txt -Lang=es -OutputFile=tts/ahotts-code/bin/temp/Output.wav -DataPath=tts/ahotts-code/bin/data_tts -Speed=100\n')
        
        # Read generated wav
        af = wave.open(r"tts/ahotts-code/bin/temp/Output.wav","rb")
        audiodata = af.readframes(self.chunk)
        
        # Play audio
        while audiodata:
            self.audiostream.write(audiodata) #Cogemos cacho y a sonar
            audiodata=af.readframes(self.chunk) #Cargamos el siguiente cacho
        
        self.audiostream.stop_stream()
                
    
    def close(self):
        
        self.audiostream.close() #Cerramos el audiostream
        self.pa.terminate() #Cerramos el reproductor
        os.system('rm tts/ahotts-code/bin/temp/input.txt tts/ahotts-code/bin/temp/Output.wav')
        


class GoogleTTS(object):
    
    def __init__(self):
        
        # samplewidth = 40
        # channels = 1
        # rate = 44100
        
        # self.pa = pyaudio.PyAudio()
        # self.audiostream = self.pa.open(format = self.pa.get_format_from_width(samplewidth),
        #                 channels = channels,
        #                 rate = rate,
        #                 output = True)
        
        self.language = 'es-es'
        
        # Starting the mixer
        mixer.init()
          
        
        
        
    def tts(self, text):
        
        myobj = gTTS(text=text, lang=self.language, slow=False)
        myobj.save("thrash/output.mp3")
        
        # Loading the song
        mixer.music.load("thrash/output.mp3")
          
        # Setting the volume
        mixer.music.set_volume(0.7)
          
        # Start playing the song
        mixer.music.play()
        
    
    def close(self):
        mixer.quit()
        

#El error handler funciona como le sale de los huevos, así que procede a ignorarlo.
#Lo puse para ocultar una serie de prints que hace el pyaudio cuando lo lanzas.
#Debería funcionar sin él


class AhoTTS2(object):
    
    def __init__(self):
        
        self.c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        
        try:
            asound = ctypes.cdll.loadLibrary('libasound.so.2')
            asound.snd_lib_error_set_handler(self.c_error_handler)
        except OSError:
            pass
        
        self.chunk = 1024
        #os.chdir('ahotts-code/bin')
    
    
    def tts(self, text):
        
        #Creamos un txt. Solo valen los charsets que aparecen en la siguiente línea
        f=open("tts/ahotts-code/bin/temp/input.txt","w+",encoding ="iso-8859-15") #cp1252 o iso-8859-1
        f.write(text) #Escribimos la entrada en consola en el fichero
        f.close() #Cerramos fichero
        
        #Llamada al tts. Los parámetros son bastante explicativos.
        os.system('./tts/ahotts-code/bin/tts -InputFile=tts/ahotts-code/bin/temp/input.txt -Lang=es -OutputFile=tts/ahotts-code/bin/temp/Output.wav -DataPath=tts/ahotts-code/bin/data_tts -Speed=100\n')
        
        
        #Abrimos el .wav con los datos a reproducir
        af = wave.open(r"tts/ahotts-code/bin/temp/Output.wav","rb")
        pa = pyaudio.PyAudio() #Creamos el reproductor
        
        #Creamos el audiostream
        audiostream = pa.open(format = pa.get_format_from_width(af.getsampwidth()),
                                channels = af.getnchannels(),
                                rate = af.getframerate(),
                                output = True,)
        
        audiodata = af.readframes(self.chunk)#Leemos el fichero en "trozos"
        
        #Bucle de reproducción hasta final de info
        while audiodata:
            audiostream.write(audiodata) #Cogemos cacho y a sonar
            audiodata=af.readframes(self.chunk) #Cargamos el siguiente cacho
        
        audiostream.stop_stream() #Paramos
        audiostream.close() #Cerramos el audiostream
        
        pa.terminate() #Cerramos el reproductor
        
        os.system('rm tts/ahotts-code/bin/temp/input.txt tts/ahotts-code/bin/temp/Output.wav')
        



def py_error_handler(filename, line, function, err, fmt):
    pass





