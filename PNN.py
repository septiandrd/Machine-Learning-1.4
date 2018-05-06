import numpy as np
np.set_printoptions(suppress=True)

def loadDataTrain(path) : # LOAD DATA TRAIN KE DALAM ARRAY
    data_train = []
    data_train_file = open(path)
    for lines in data_train_file:
        num_list = []
        for number in lines.strip().split():
            num_list.append(float(number))
        num_list = np.array(num_list)
        data_train.append(num_list)
    return data_train

def loadDataTest(path) : # LOAD DATA TEST KE DALAM ARRAY
    data_test = []
    data_test_file = open(path)
    for lines in data_test_file :
        num_list = []
        for number in lines.strip().split() :
            num_list.append(float(number))
        data_test.append(num_list)
    return data_test

if __name__ == '__main__':

    data_train = np.array(loadDataTrain('dataset/data_train_100_a.txt')) # LOAD DATA TRAIN
    data_test = np.array(loadDataTest('dataset/data_valid_50_a.txt'))   # LOAD DATA VALIDASI

    a = 0.8

    CLASS = [0,1,2]
    CLASS_COUNT = len(CLASS)
    FEATURE_COUNT = 3
    SIGMA = a
    CORRECT_COUNT = 0

    TEST_OUTPUT = []

    for test_data in data_test :

        SUM_RESULT = []
        TEST_DATA_OUTPUT = []
        TEST_DATA_OUTPUT.append(test_data[0])
        TEST_DATA_OUTPUT.append(test_data[1])
        TEST_DATA_OUTPUT.append(test_data[2])

        for i in range (CLASS_COUNT) :

            train_data = data_train[data_train[:,3]==i]
            DATA_IN_CLASS_K = len(train_data)
            SUM_CLASS = 0.0

            for j in range(DATA_IN_CLASS_K) :
                gx = np.exp((((np.power(test_data[0]-train_data[j][0],2)+np.power(test_data[1]-train_data[j][1],2)+np.power(test_data[2]-train_data[j][2],2))*-1)/(2*pow(SIGMA,2))))
                SUM_CLASS += gx

            SUM_RESULT.append(SUM_CLASS)

        if SUM_RESULT.index(max(SUM_RESULT)) == int(test_data[3]) :
            CORRECT_COUNT += 1

        TEST_DATA_OUTPUT.append(SUM_RESULT.index(max(SUM_RESULT)))
        TEST_OUTPUT.append(TEST_DATA_OUTPUT)

    print("Sigma : ",round(SIGMA,2)," Akurasi : ",(CORRECT_COUNT/50)*100,"%")
    TEST_OUTPUT = np.array(TEST_OUTPUT)

    a+=1