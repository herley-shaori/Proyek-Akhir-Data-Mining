
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm

class Klas:
	def __init__(self):
		self.data = pandas.read_csv('hog.csv')
		self.labels = ['bubur_ayam','gado_gado','kerak_telor', 'ketoprak', 'kue_cincin', 'kue_rangi','opor_ayam','pindang_bandeng','roti_buaya','soto_betawi','tumis_peda']
		
	def trial(self):
		col = []
		for x in range(768):
			col.append(str(x))

		X = self.data[col]
		y = self.data['label']

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

		print("Memulai Training SVM")
		clf = svm.SVC(gamma=0.001, decision_function_shape='ovo')
		# clf = svm.LinearSVC()
		clf.fit(X_train,y_train)
		print("Training SVM selesai \n")

		y_pred = clf.predict(X_test)

		print(y_pred)
		print(y_pred.shape)
		print(y_test.shape)

		print(confusion_matrix(y_test,y_pred))  
		print(classification_report(y_test,y_pred))
		print("Jumlah Data Latih: ", X_train.shape[0])
		print("Jumlah Data Uji: ", X_test.shape[0])

		# ---------------------------------------------
		# data_kelas_0 = None
		# for x in range(10):
		# 	if(data_kelas_0 is None):
		# 		data_kelas_0 = self.data.iloc[[x]]
		# 	else:
		# 		data_kelas_0 = data_kelas_0.append(self.data.iloc[[x]])

		# label_0 = []
		# for x in range(data_kelas_0.shape[0]):
		# 	label_0.append(0)

		# data_kelas_0 =data_kelas_0.drop(columns=['label'])
		# data_kelas_0['label'] = label_0
		
		# data_kelas_1 = None
		# for x in range(10, self.data.shape[0]):
		# 	if(data_kelas_1 is None):
		# 		data_kelas_1 = self.data.iloc[[x]]
		# 	else:
		# 		data_kelas_1 = data_kelas_1.append(self.data.iloc[[x]])

		# label_0 = []
		# for x in range(data_kelas_1.shape[0]):
		# 	label_0.append(1)

		# data_kelas_1 = data_kelas_1.drop(columns=['label'])
		# data_kelas_1['label'] = label_0

		# col = []
		# for x in range(768):
		# 	col.append(str(x))

		# joint_data_example = data_kelas_0.append(data_kelas_1)

		# X = joint_data_example[col]
		# y = joint_data_example['label']

		# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

		# print(X_train.shape)

		# print("Memulai Training SVM")
		# clf = svm.SVC()
		# clf.fit(X_train,y_train)
		# print("Training SVM selesai \n")

		# y_pred = clf.predict(X_test)
		# print(y_pred)
		# print(confusion_matrix(y_test,y_pred))  
		# print(classification_report(y_test,y_pred))  
