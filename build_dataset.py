import cv2
import os
import pandas
from sklearn.utils import shuffle


path = "Q1_TrainingData/"
folder_list = os.listdir(path)
print(folder_list)

df = pandas.DataFrame(columns = ['filename', 'label'])

for i in range(len(folder_list)):

	if folder_list[i] != '.DS_Store':
		sub_fol_list = os.listdir(path+folder_list[i])
		# print(path + folder_list[i])
		
		for images in sub_fol_list:
			img_path = path + folder_list[i] + "/" + images
			# print(img_path)['AamirKhan', 'SalmanKhan', 'ShahrukhKhan']
			img = cv2.imread(img_path,0)
			resize_img = cv2.resize(img, (64,64))

			if folder_list[i] == 'AamirKhan':
				img_name = "1" + images
				df = df.append({'filename' : img_name, 'label' : 1}, ignore_index = True)
				cv2.imwrite("testdata/" + img_name,resize_img)
			if folder_list[i] == 'SalmanKhan':
				img_name = "2" + images
				df = df.append({'filename' : img_name, 'label' : 2}, ignore_index = True)
				cv2.imwrite("testdata/" + img_name,resize_img)
			if folder_list[i] == 'ShahrukhKhan':
				img_name = "0" + images
				df = df.append({'filename' : img_name, 'label' : 0}, ignore_index = True)
				cv2.imwrite("testdata/" + img_name,resize_img)
				# print(resize_img.shape)
				# cv2.imwrite("testdata/")
df = shuffle(df)
df.to_csv("dataset.csv")
print(df.head())

