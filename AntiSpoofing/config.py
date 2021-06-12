import os
import sys
Models_paths = {
    "ML-Print":os.getcwd()+"/AntiSpoofing/Pretrain/print-attack_ycrcb_luv_extraTreesClassifier.pkl",
    "ML-Replay":os.getcwd()+"/AntiSpoofing/Pretrain/replay-attack_ycrcb_luv_extraTreesClassifier.pkl",
    "DeNoise":os.getcwd()+"/AntiSpoofing/Pretrain/DeNoise",
    "FaceBagNet32":os.getcwd()+"/AntiSpoofing/Pretrain/FaceBagNet32.pth",
    "FaceBagNet48":os.getcwd()+"/AntiSpoofing/Pretrain/FaceBagNet48.pth",
    "FaceBagNet64":os.getcwd()+"/AntiSpoofing/Pretrain/FaceBagNet64.pth",
    "HyperFAS":os.getcwd()+"/AntiSpoofing/Pretrain/HyperFAS.h5",
    "EfficientB3":os.getcwd()+"/background/AntiSpoofing/Pretrain/b3.pth.tar"
    }
