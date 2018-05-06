from sklearn.neural_network import MLPClassifier
import Dataset

datatrain = Dataset.loadDataTrain()

x = datatrain[:100,:3]
y = datatrain[:100,3]

MLP = MLPClassifier()

MLP.fit(x,y)

prediksi = MLP.predict(datatrain[100:,:3])

print('Akurasi :',Dataset.akurasi(prediksi),'%')