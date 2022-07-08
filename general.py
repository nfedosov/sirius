from model import Model
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from pylsl.pylsl import lib, StreamInfo, FOREVER, c_int, c_double, byref, handle_error, StreamOutlet
import xml.etree.ElementTree as ET
from lsl_inlet import LSLInlet
import time
import pynput

from pynput.keyboard import Key, Controller




class General:
    def __init__(self):

        #self.path_dir = 'results/session_{}/'.format(session_num)
        pass
    
    
    
  

           


    def run_game(self):
      
        inlet = LSLInlet('real_time')
        fs = int(inlet.get_frequency())
        channels = inlet.get_channels_labels()
        n_channels = len(channels)
        
        n_samples_received = 0
        
        model = Model()
        model.load_model()
        
        
        time.sleep(10)
        keyboard = Controller()
        key = "s"
        keyboard.press(key)
        keyboard.release(key)
        
        while(True):
            chunk, t_stamp = inlet.get_next_chunk()
            if chunk is not None:
                model.predict_once(chunk)        
                
        


model = Model()

model.fit()
model.test()

#general = General()
#general.run_game()




