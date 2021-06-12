import keras
from keras.layers import  *
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
import glob
import timeit
class HyperFAS:
  '''
  Repo: https://github.com/zeusees/HyperFAS.git
  input : BGR
  output : 0/1 => S/R
  
  '''
  def __init__(self,pretrain):
    self.model = load_model(pretrain,compile=False)
    self.mode = "BGR"
    self.name = "HyperFAS"
    print("[INFO] : Loaded HyperFAS")  

  def predict(self,img_batch,threshold = 0.95):
    preds_list = []
    for imag in img_batch:
      X = (cv2.resize(imag,(224,224))-127.5)/127.5
      X = np.array([X])
      score = self.model.predict(X)[0]
      if score > threshold:
        preds_list.append(1) # Real
      else:
        preds_list.append(0) # Spoof
    return preds_list

      
