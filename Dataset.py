import numpy as np

def loadDataTrain() : # LOAD DATA TRAIN KE DALAM ARRAY
    data_train = []
    data_train_file = open('dataset/data_train_PNN.txt')
    for lines in data_train_file:
        num_list = []
        for number in lines.strip().split():
            num_list.append(float(number))
        num_list = np.array(num_list)
        data_train.append(num_list)
    data_train = np.array(data_train)
    return data_train

def loadDataTest() : # LOAD DATA TEST KE DALAM ARRAY
    data_test = []
    data_test_file = open('dataset/data_test_PNN.txt')
    for lines in data_test_file :
        num_list = []
        for number in lines.strip().split() :
            num_list.append(float(number))
        data_test.append(num_list)
    data_test = np.array(data_test)
    return data_test

def loadDataValidasi() : # LOAD DATA TEST KE DALAM ARRAY
    data_test = []
    data_test_file = open('dataset/data_test_PNN.txt')
    for lines in data_test_file :
        num_list = []
        for number in lines.strip().split() :
            num_list.append(float(number))
        data_test.append(num_list)
    data_test = np.array(data_test)
    return data_test

def akurasi(prediksi) :
    dataTrain = loadDataTrain()
    correct = 0
    for i,j in zip(range(100,149),range(len(prediksi))) :
        if dataTrain[i][3] == prediksi[j] :
            correct +=1
    return correct/50*100