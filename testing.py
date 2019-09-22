from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
import time
import pandas as pd


def testing_image():

	path = "Q1_TestData/"
	folder = os.listdir(path)
	df = pd.DataFrame(columns = ['filename', 'test_label'])
	print("[INFO] loading network...")
	model=load_model('face_reg.h5')

	for images in folder:
		print (images)

		# load the image
		image = cv2.imread(path + images,0)
		# orig = image.copy()
		orig = cv2.imread(path + images,1)
		 
		# pre-process the image for classification
		image = cv2.resize(image, (64, 64))
		image = image.astype("float32") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

	
		predictions=model.predict(image)
		print(predictions[0])
		label = [0, 1, 2]

		label_zero = "{}: {:.2f}%".format(label[0], predictions[0][0] * 100)
		label_one = "{}: {:.2f}%".format(label[1], predictions[0][1] * 100)
		label_two = "{}: {:.2f}%".format(label[2], predictions[0][2] * 100)

		predicted_res = max(predictions[0])
		predicted_label = list(predictions[0]).index(predicted_res)
		
		df = df.append({'filename' : images, 'test_label' : predicted_label}, ignore_index = True)

	df.to_csv("TestingData_Result.csv")


testing_image()