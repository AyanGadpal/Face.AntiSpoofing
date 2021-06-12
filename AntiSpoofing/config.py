import os

PATH_TO_PRETRAIN = os.path.join(os.getcwd(),"AntiSpoofing","Pretrain")

def set_path(model_name):
  return os.path.join(PATH_TO_PRETRAIN,model_name)


Models_paths = {
    "ML-Print":set_path("print-attack_ycrcb_luv_extraTreesClassifier.pkl"),
    "ML-Replay":set_path("replay-attack_ycrcb_luv_extraTreesClassifier.pkl"),
    "DeNoise":set_path("DeNoise"),
    "FaceBagNet32":set_path("FaceBagNet32.pth"),
    "FaceBagNet48":set_path("FaceBagNet48.pth"),
    "FaceBagNet64":set_path("FaceBagNet64.pth"),
    "HyperFAS":set_path("HyperFAS.h5"),
    "EfficientB3":set_path("b3.pth.tar")
    }




