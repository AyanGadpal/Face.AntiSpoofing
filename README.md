# Face Anti Spoofing 
With the ascent of Face recognition technology being used in the high-security area, The need for face antispoofing is increasing, Thus this repository contains various pre-trained models, This Repository attempts to bring all those models together and create an easy to use API for them.

# Usage
Import AntiSpoofing from AntiSpoofing, Create an object and use it
```bash
obj = AntiSpoofing()
```

Use AntiSpoofing.PrintModelChoice() to get all available object. 

```bash
AntiSpoofing.PrintModelChoice() 
```
Use obj.setModel to select model to use. Each model has accuracy performance trade-off. 

# References 
1. Shen, Tao & Huang, Yuyu. (2019). FaceBagNet: Bag-Of-Local-Features Model for Multi-Modal Face Anti-Spoofing. 10.1109/CVPRW.2019.00203.
2. [HyperFAS](https://spotvisionai.streamlit.app/)

