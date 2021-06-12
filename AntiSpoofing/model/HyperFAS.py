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

  def predict(self,imag,threshold = 0.95):
    X = (cv2.resize(imag,(224,224))-127.5)/127.5
    X = np.array([X])
    tic = timeit.default_timer()
    score = self.model.predict(X)[0]
    toc = timeit.default_timer()
    print("[INFERENCE TIME] : ",toc-tic)

    if score > threshold:
      return 1
    else:
      return 0
