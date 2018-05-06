from sklearn.svm import SVC
import Dataset

datatrain = Dataset.loadDataTrain()

x = datatrain[:100,:3]
y = datatrain[:100,3]

SVM = SVC()

SVM.fit(x,y)

prediksi = SVM.predict(datatrain[100:,:3])

print('Akurasi :',Dataset.akurasi(prediksi),'%')