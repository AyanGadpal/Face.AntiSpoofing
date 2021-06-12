import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import glob
import timeit

class DeNoise:
  '''
  DL model, Or semi DL
  It decides spoofiness based on the noise on the image
  Repo : https://github.com/yaojieliu/ECCV2018-FaceDeSpoofing.git
  Image Mode : BGR
  Output : Score b/t -1 to 1, higher (--->1) represent spoofiness
  # USES TF version 1
  '''
  def __init__(self,Path,Gpu=0.2):
    self.Path = Path
    self.mode = "BGR"
    self.name = "DeNoise"
    self.starter = True
    self.graph=tf.Graph()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=Gpu)
    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=self.graph)
    tf.saved_model.loader.load(self.sess, 
          [tf.saved_model.tag_constants.SERVING], 
          self.Path)
    with self.graph.as_default():
      self.image  = tf.get_default_graph().get_tensor_by_name("input:0")
      self.scores = tf.get_default_graph().get_tensor_by_name("Mean_2:0")

  def predict(self,img_batch,threshold = 0.45):
    preds = []
    for img in img_batch:
      sc = self.sess.run(self.scores,feed_dict={self.image : img})
      if sc <= threshold:
        preds.append(1)
      else:
        preds.append(0)
    return preds

  # def predict_v1(self,img,threshold = 0.45):
  #     import tensorflow.compat.v1 as tf

  #     inputname = "input:0"
  #     outputname = "Mean_2:0"

  #     self.image  = tf.get_default_graph().get_tensor_by_name("input:0")
  #     self.scores = tf.get_default_graph().get_tensor_by_name("Mean_2:0")
  #     with tf.Session() as sess:
  #     # load the facepad model
  #       tf.saved_model.loader.load(sess, 
  #             [tf.saved_model.tag_constants.SERVING], 
  #             self.Path)
        
  #       frame = cv2.imread(f)
  #       sc = sess.run(self.scores,feed_dict={self.image : frame})
  #       # print(sc)
  #       if sc <= threshold:
  #         print(1)
  #       else:
  #         print(0)
      
  #       return (real,spoof)