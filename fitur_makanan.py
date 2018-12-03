import glob
import cv2
import numpy
import pandas
# import matplotlib.pyplot as plt

class Fitur:
	def __init__(self):
		self.labels = ['bubur_ayam','gado_gado','kerak_telor', 'ketoprak', 'kue_cincin', 'kue_rangi','opor_ayam','pindang_bandeng','roti_buaya','soto_betawi','tumis_peda']
		per_image_label = []

		self.images_in_all_folder = []
		self.largest_width = 0
		self.largest_height = 0

		nomor_label = 0
		for x in self.labels:
			images_in_one_folder = [cv2.imread(file) for file in glob.glob('data_full/'+x+'/*.jpg')]
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
		# -----------------------------------------------------
		# self.labels = ['bubur_ayam','gado_gado','kerak_telor', 'ketoprak', 'kue_cincin', 'kue_rangi','opor_ayam','pindang_bandeng','roti_buaya','soto_betawi','tumis_peda']
		# per_image_label = []

		# self.images_in_all_folder = []
        
		# nomor_label = 0
		# for x in self.labels:
		# 	images_in_one_folder = [cv2.imread(file) for file in glob.glob('data/'+x+'/*.jpg')]
		# 	for i in range(100):
		# 		per_image_label.append(nomor_label)
		# 	self.images_in_all_folder.append(images_in_one_folder)
		# 	nomor_label = nomor_label + 1
            
		# a = self.images_in_all_folder[0]
		# b = a[0]
		# img = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
		# plt.imshow(img)        
		# cv2.imshow('image',b)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		self.hog_extraction(per_image_label)


	def array_fill(self, data):
		nol = numpy.zeros((3000,3000))

		for baris in range(data.shape[0]):
			for kolom in range(data.shape[1]):
				elemen = data[baris][kolom]
				nol[baris][kolom] = elemen

		return nol

	def color_frequency(self, red,green,blue):
		a = numpy.zeros(256,dtype='int')
		b = numpy.zeros(256,dtype='int')
		c = numpy.zeros(256,dtype='int')

		red = red.astype('int')

		for baris in range(red.shape[0]):
			a[baris] = red[baris][1]

		for baris in range(green.shape[0]):
			b[baris] = green[baris][1]

		for baris in range(blue.shape[0]):
			c[baris] = blue[baris][1]

		a = numpy.vstack((a,b))
		a = numpy.vstack((a,c)).ravel()

		return a

	def color_frequency_normalization(self, biggest_frequency, all_to_be_normalized):

		haha = None

		for x in range(len(all_to_be_normalized)):
			data_awal = all_to_be_normalized[x]
			data_awal = numpy.true_divide(data_awal,biggest_frequency)
			all_to_be_normalized[x] = data_awal.astype('float64')
			if(haha is None):
				haha = data_awal
			else:
				haha = numpy.vstack((haha,data_awal))

		return haha

	def hog_extraction(self,per_image_label):
		red = 0
		green = 0
		blue = 0

		x_index = 0

		removed = []
		# Mencari nilai terbesar.
		for x in self.images_in_all_folder:
			# print("X: ", x_index)

			y_index = 0
			for y in x:
				# print("Y", y_index)
				# y_index = y_index + 1
				if(len(cv2.split(y)) == 3):
					b, g, r = cv2.split(y) # For BGR image
					bval = numpy.amax(b)
					gval = numpy.amax(g)
					rval = numpy.amax(r)

					if(rval > red):
						red = rval
					if(gval > green):
						green = gval
					if(bval > blue):
						blue = bval
				else:
					removed.append(x_index)

			x_index = x_index + 1

		# Normalisasi.
		all_to_be_normalized = []

		biggest_frequency = 0
		for x in self.images_in_all_folder:
			for y in x:
				if(len(cv2.split(y)) == 3):
					b, g, r = cv2.split(y) # For BGR image

					ur,cr = numpy.unique(r, return_counts=True)
					ug,cg = numpy.unique(g, return_counts=True)
					ub,cb = numpy.unique(b, return_counts=True)

					a = self.color_frequency(numpy.asarray((ur, cr)).T,numpy.asarray((ug, cg)).T,numpy.asarray((ub, cb)).T)

					if(numpy.amax(a) > biggest_frequency):
						biggest_frequency = numpy.amax(a)

					all_to_be_normalized.append(a)

		all_to_be_normalized = self.color_frequency_normalization(biggest_frequency,all_to_be_normalized)

		col = []
		for x in range(768):
			col.append(str(x))

		# print(col)

		# Membuat DataFrame.
		pandas_data = pandas.DataFrame(all_to_be_normalized,columns=col)

		# Point removal.
		index_to_be_removed = []
		for x in removed:
			for y in range(len(per_image_label)):
				if(x == per_image_label[y]):
					index_to_be_removed.append(y)
					break

		for x in index_to_be_removed:
			del per_image_label[x]

		# print("Index Removed: ", index_to_be_removed)
		# print("Label Length: ", len(per_image_label))
		# print("Pandas Length: ", pandas_data.shape[0])

		pandas_data['label'] = per_image_label

		# Save Point.
		pandas_data.to_csv('hog.csv', index=False)