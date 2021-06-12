import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from AntiSpoofing.model.FaceBag import FaceBagNet
from AntiSpoofing.model.HyperFAS import HyperFAS 
from AntiSpoofing.model.ML_Model import ML_Model 
from AntiSpoofing.model.DeNoise import DeNoise
from AntiSpoofing.model.EfficientNet import EfficientNet

from AntiSpoofing.config import Models_paths

import cv2
import numpy as np
import glob

class AntiSpoofing:
  """
  Model:
  0 is Spoof
  1 is Real
  """
  def __init__(self,ModelName=""):
    import sys
    sys.path.append("/content/drive/My Drive/AntiSpoofing")
    self.model = None
    self.mode = "BGR" # Default 
    self.ModelName = ModelName
  

  def lable(self,n):
    if n == 0:
      return "Spoof" # 0
    else:
      return "Real" # 1

  def setModel(self,ModelName,Gpu=0.2):
    self.ModelName = ModelName
    if ModelName in Models_paths:
      
      if ModelName[:2] == "ML":
        print("[INFO] : ML MODEL")
        self.model = ML_Model(Models_paths[ModelName])

      elif ModelName == "DeNoise":
        print("[INFO] : Getting DeNoise Model")
        self.model = DeNoise(Models_paths[ModelName],Gpu=Gpu)
     
      elif ModelName[:10] == "FaceBagNet":
        print("[INFO] : Getting FaceBagNet")
        self.model = FaceBagNet(Models_paths[ModelName])
      
      elif ModelName == "HyperFAS":
        print("[INFO] : Getting HyperFAS")
        self.model = HyperFAS(Models_paths[ModelName])
      
      elif ModelName == "EfficientB3":
        print("[INFO] : Getting EfficientB3")
        self.model = EfficientNet(Models_paths[ModelName],b=3)
      
      self.mode = self.model.mode
      print("[INFO] : Model Selected : ",self.model.name)
    
    else:
      print("[ERROR] Model Not Found !")
    
  def predict(self,imgPath):
    # Face Detection 
    face = None
    if self.mode == "BGR" or self.mode == "BGR-MASK":
      if isinstance(imgPath,str):
        print("[ERROR] :"+self.ModelName+" Need Image NPARRAY")
        return
      # Pass to model predict
      prediction = self.model.predict(imgPath)
      print(self.lable(prediction))

  @staticmethod
  def PrintModelChoice():
    print("Here are the name code for the models")
    for i,pair in enumerate(Models_paths):
      print(i+1,") ",pair)