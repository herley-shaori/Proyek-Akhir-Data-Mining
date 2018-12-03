
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn import tree
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

class Klas:
	def __init__(self):
		self.data = pandas.read_csv('hog.csv')
		self.labels = ['bubur_ayam','gado_gado','kerak_telor', 'ketoprak', 'kue_cincin', 'kue_rangi','opor_ayam','pindang_bandeng','roti_buaya','soto_betawi','tumis_peda']

	def ada_boost_trial(self, with_pca=False):
		print("ADA Boost")
		col = []
		for x in range(768):
			col.append(str(x))

		X = self.data[col]
		y = self.data['label']


		if(with_pca):
			print("PCA TRUE")
			# pca = PCA(n_components=40)
			pca = PCA(0.98)
			# pca = PCA(0.7)
			X = pca.fit_transform(X)

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

			clf = AdaBoostClassifier(n_estimators=100)
			clf.fit(X_train,y_train)
			y_pred = clf.predict(X_test)

			print(y_pred.shape)
			print(y_test.shape)

			print(confusion_matrix(y_test,y_pred))  
			print(classification_report(y_test,y_pred))
			print("Jumlah Data Latih: ", X_train.shape[0])
			print("Jumlah Data Uji: ", X_test.shape[0])
			print("Jumlah Fitur setelah PCA: ", X.shape)
		else:
			print("PCA FALSE")
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

			clf = AdaBoostClassifier()
			clf.fit(X_train,y_train)
			y_pred = clf.predict(X_test)

			print(y_pred.shape)
			print(y_test.shape)

			print(confusion_matrix(y_test,y_pred))  
			print(classification_report(y_test,y_pred))
			print("Jumlah Data Latih: ", X_train.shape[0])
			print("Jumlah Data Uji: ", X_test.shape[0])

	def extra_trees_trial(self, with_pca=False):
		print("Extra Trees")
		col = []
		for x in range(768):
			col.append(str(x))

		X = self.data[col]
		y = self.data['label']


		if(with_pca):
			print("PCA TRUE")
			# pca = PCA(n_components=40)
			pca = PCA(0.98)
			# pca = PCA(0.7)
			X = pca.fit_transform(X)

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

			clf = ExtraTreesClassifier(n_estimators=500)
			clf.fit(X_train,y_train)
			y_pred = clf.predict(X_test)

			print(y_pred.shape)
			print(y_test.shape)

			print(confusion_matrix(y_test,y_pred))  
			print(classification_report(y_test,y_pred))
			print("Jumlah Data Latih: ", X_train.shape[0])
			print("Jumlah Data Uji: ", X_test.shape[0])
			print("Jumlah Fitur setelah PCA: ", X.shape)
		else:
			print("PCA FALSE")
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

			clf = ExtraTreesClassifier(n_estimators=100)
			clf.fit(X_train,y_train)
			y_pred = clf.predict(X_test)

			print(y_pred.shape)
			print(y_test.shape)

			print(confusion_matrix(y_test,y_pred))  
			print(classification_report(y_test,y_pred))
			print("Jumlah Data Latih: ", X_train.shape[0])
			print("Jumlah Data Uji: ", X_test.shape[0])

	def random_forest_trial(self,with_pca=False):
		print("Random Forest")
		col = []
		for x in range(768):
			col.append(str(x))

		X = self.data[col]
		y = self.data['label']


		if(with_pca):
			print("PCA TRUE")
			# pca = PCA(n_components=40)
			pca = PCA(0.98)
			# pca = PCA(0.7)
			X = pca.fit_transform(X)

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

			clf = RandomForestClassifier(n_estimators=500)
			clf.fit(X_train,y_train)
			y_pred = clf.predict(X_test)

			print(y_pred.shape)
			print(y_test.shape)

			print(confusion_matrix(y_test,y_pred))  
			print(classification_report(y_test,y_pred))
			print("Jumlah Data Latih: ", X_train.shape[0])
			print("Jumlah Data Uji: ", X_test.shape[0])
			print("Jumlah Fitur setelah PCA: ", X.shape)
		else:
			print("PCA FALSE")
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

			clf = RandomForestClassifier(n_estimators=100)
			clf.fit(X_train,y_train)
			y_pred = clf.predict(X_test)

			print(y_pred.shape)
			print(y_test.shape)

			print(confusion_matrix(y_test,y_pred))  
			print(classification_report(y_test,y_pred))
			print("Jumlah Data Latih: ", X_train.shape[0])
			print("Jumlah Data Uji: ", X_test.shape[0])


	def xgb_trial(self,with_pca=False):
		print("XGB Trial")
		col = []
		for x in range(768):
			col.append(str(x))

		X = self.data[col]
		y = self.data['label']


		if(with_pca):
			print("PCA TRUE")
			# pca = PCA(n_components=40)
			pca = PCA(0.99)
			# pca = PCA(0.7)
			X = pca.fit_transform(X)

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

			clf = XGBClassifier()
			clf.fit(X_train,y_train)
			y_pred = clf.predict(X_test)

			print(y_pred.shape)
			print(y_test.shape)

			print(confusion_matrix(y_test,y_pred))  
			print(classification_report(y_test,y_pred))
			print("Jumlah Data Latih: ", X_train.shape[0])
			print("Jumlah Data Uji: ", X_test.shape[0])
			print("Jumlah Fitur setelah PCA: ", X.shape)
		else:
			print("PCA FALSE")
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

			clf = XGBClassifier()
			clf.fit(X_train,y_train)
			y_pred = clf.predict(X_test)

			print(y_pred.shape)
			print(y_test.shape)

			print(confusion_matrix(y_test,y_pred))  
			print(classification_report(y_test,y_pred))
			print("Jumlah Data Latih: ", X_train.shape[0])
			print("Jumlah Data Uji: ", X_test.shape[0])

	def pca_svm_trial(self):
		print("SVM-PCA Trial")
		col = []
		for x in range(768):
			col.append(str(x))

		X = self.data[col]
		y = self.data['label']

		# pca = PCA(n_components=40)
		pca = PCA(0.98)
		# pca = PCA(0.7)
		X = pca.fit_transform(X)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

		print("Memulai Training SVM")
		# clf = svm.SVC(gamma=0.001, decision_function_shape='ovo')
		clf = svm.LinearSVC()
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
		print("Jumlah Fitur setelah PCA: ", X.shape)


	def pca_dt_trial(self):
		col = []
		for x in range(768):
			col.append(str(x))

		X = self.data[col]
		y = self.data['label']

		# pca = PCA(n_components=40)
		# pca = PCA(0.98)
		pca = PCA(0.7)
		X = pca.fit_transform(X)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)

		print(confusion_matrix(y_test,y_pred))  
		print(classification_report(y_test,y_pred))
		print("Jumlah Fitur setelah PCA: ", X.shape)
		print("Jumlah Data Latih: ", X_train.shape[0])
		print("Jumlah Data Uji: ", X_test.shape[0]) 

	def dt_trial(self):
		col = []
		for x in range(768):
			col.append(str(x))

		X = self.data[col]
		y = self.data['label']

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)

		print(confusion_matrix(y_test,y_pred))  
		print(classification_report(y_test,y_pred))
		print("Jumlah Data Latih: ", X_train.shape[0])
		print("Jumlah Data Uji: ", X_test.shape[0]) 
		
	def svm_trial(self):
		col = []
		for x in range(768):
			col.append(str(x))

		X = self.data[col]
		y = self.data['label']

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=None)

		print("Memulai Training SVM")
		# clf = svm.SVC(gamma=0.001, decision_function_shape='ovo')
		clf = svm.LinearSVC()
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
