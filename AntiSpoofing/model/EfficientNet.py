from timm.data.transforms_factory import create_transform
from timm.models import create_model
import torch
import numpy as np
import cv2
import glob
from PIL import Image

class EfficientNet:
  def __init__(self,pretrain,b=0):
    self.mode = "BGR"
    self.name = "EfficientNetB"+str(b)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = create_model(
      "tf_efficientnet_b"+str(b)+"_ns",
      num_classes=2,
      pretrained=False,
      checkpoint_path=pretrain)
      
    self.model.eval()
    self.model = self.model.to(self.device)
    print("[INFO] : Loaded Efficient Net B",b)

    self.preprocessor = create_transform(
      input_size=112,
      tf_preprocessing=False,
      mean=(0.485, 0.456, 0.406),
      std=(0.229, 0.224, 0.225),
      crop_pct=0.949
      )  
  
  def preprocess(self,cvimg):
    img = Image.fromarray(cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB))
    input = self.preprocessor(img)
    return input.to(self.device)

  def process(self,batch):
    labels = self.model(batch)
    output = torch.nn.functional.softmax(labels)
    output = torch.argmin(output, dim=1)
    return output

  def postprocess(self,pretrain):
    pass
  