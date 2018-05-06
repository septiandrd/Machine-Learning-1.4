from sklearn.naive_bayes import GaussianNB
import Dataset

datatrain = Dataset.loadDataTrain()

x = datatrain[:100,:3]
y = datatrain[:100,3]

model = GaussianNB()
model.fit(x,y)

prediksi = model.predict(datatrain[100:,:3])

print('Akurasi :',Dataset.akurasi(prediksi),'%')