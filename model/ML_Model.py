from sklearn.externals import *
import joblib
import warnings
import numpy as np
import cv2
import glob
import timeit

class ML_Model:
  '''
  ML based Models
  Repo : https://github.com/ee09115/spoofing_detection
  Image Mode : BGR
  '''

  def __init__(self,Path):
    from sklearn.externals import joblib
    import warnings

    # Ignore the unwanted warning
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)
      # Load the model
      self.model = joblib.load(Path)

    # MODE BGR
    self.mode = "BGR"
    self.name = Path.split("/")[-1]
    
    print("[INFO] : Successfully Loaded :",Path.split("/")[-1])

  def preprocess(self,img):

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

    ycrcb_hist = self.calc_hist(img_ycrcb)
    luv_hist = self.calc_hist(img_luv)

    feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
    return feature_vector

  # Pass the list of "feature_vector"
  def process(self,feature_vectors):
    prediction = self.model.predict_proba(feature_vectors)
    preds = np.argmax(prediction, axis=1) 
    return preds
  
  # No Preproccessing is involed
  def postprocess(self,preds):
    pass

  def calc_hist(self,img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)
  
  def predict(self,img_batch, threshold=0.7):
    pred_list = []

    for img in img_batch:
      img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
      img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

      ycrcb_hist = self.calc_hist(img_ycrcb)
      luv_hist = self.calc_hist(img_luv)

      feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
      feature_vector = feature_vector.reshape(1, len(feature_vector))
      prediction = self.model.predict_proba(feature_vector)
      
      prob = prediction[0][1]

      if prob >= threshold:
        pred_list.append(0) # Spoof
      else:
        pred_list.append(1) # Real
    return pred_list