import glob
import cv2
import numpy
import pandas

class CF:
	def __init__(self):
		self.labels = ['bubur_ayam','gado_gado','kerak_telor', 'ketoprak', 'kue_cincin', 'kue_rangi','opor_ayam','pindang_bandeng','roti_buaya','soto_betawi','tumis_peda']
		per_image_label = []

		self.images_in_all_folder = []
		self.largest_width = 0
		self.largest_height = 0

		nomor_label = 0
		for x in self.labels:
			images_in_one_folder = [cv2.imread(file) for file in glob.glob('data_x/'+x+'/*.jpg')]
			cropped_images = []
			for img in images_in_one_folder:
				if(img is not None and img.shape[2] == 3):				
					height, width, channels = img.shape
					
					upper_left = (int(width / 4), int(height / 4))
					bottom_right = (int(width * 3 / 4), int(height * 3 / 4))
					rect_img = img[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]

					if(width > self.largest_width):
						self.largest_width = width

					if(height > self.largest_height):
						self.largest_height = height
					cropped_images.append(img)
					per_image_label.append(nomor_label)
			self.images_in_all_folder.append(cropped_images)
			nomor_label = nomor_label + 1
