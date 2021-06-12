from .FaceBagNetOrignal.FaceBagNet_model_A import Net
import cv2
import torch.nn.functional as F
import numpy as np
import torch
from torchvision.transforms import ToTensor
class FaceBagNet(Net):
  """
  Repo : https://github.com/ledduy610/CVPR19-Face-Anti-spoofing/tree/2ceba5d2e2587159e1cb69dbda557bb0ba40551c
  Input : BGR MASK (112,112,3)
  Output : 0/1 -> S/R
  """

  def __init__(self,pretrain):
    super().__init__(num_class=2,is_first_bn=True)
    self.load_pretrain(pretrain)
    self.mode = "BGR-MASK"
    self.eval() # Set to Testing Mode

  def predict(self,image):
    image = cv2.resize(image,(112,112))
    image = image.astype(np.float32)
    image = image / 255.00 # Normalize 
    input = ToTensor()(image) # PyTorch ke nakhare
    input = np.expand_dims(input, axis=0) # Add n as 1st of 4d input
    input = torch.FloatTensor(input)
    with torch.no_grad():
      b = 1
      n,c,w,h = input.size() # c=channel, w=width and h=height : Default 3,112,112 BGR
      self.to('cuda')
      input = input.to('cuda')
      logit,logit_id, fea   = self.__call__(input)
      logit = logit.view(b,n,2)
      logit = torch.mean(logit, dim = 1, keepdim = False)
      prob = F.softmax(logit, 1)
      prob = prob.data.cpu().numpy()
    if prob[0][0] > 0.7:
      return 0 #Spoof
    else:
      return 1 #Real
