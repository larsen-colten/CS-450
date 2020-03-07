from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class Car:
    def __init__(self):
        self.data = []
        self.targets = []
    def load_data(self):
        # Read the file
        data = pd.read_csv('03_prove/data/car.data', names=[0, 1, 2, 3, 4, 5, 'target'])

        # Convert String values into integers
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(data['target'])
        x_0 = le.fit_transform(data[0])
        x_1 = le.fit_transform(data[1])
        x_2 = le.fit_transform(data[2])
        x_3 = le.fit_transform(data[3])
        x_4 = le.fit_transform(data[4])
        x_5 = le.fit_transform(data[5])

        data['target'] = targets
        data[0] = x_0
        data[1] = x_1
        data[2] = x_2
        data[3] = x_3
        data[4] = x_4
        data[5] = x_5
  
        # Check for messy data
        for i in range(data.shape[1] - 1):
            data = data[data[i] != '']
            data = data[data[i] != '?']
            data = data[data[i] != None]

        self.data = data[[0, 1, 2, 3, 4, 5]]
        self.targets = data[['target']]

    def split_data(self, size):
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.targets, test_size=size)
        return ( x_train, x_test, y_train, y_test )

class Mpg:
    def __init__(self):
        self.data = []
        self.targets = []
    def load_data(self):
        # Read the file
        data = pd.read_csv('03_prove/data/auto-mpg.data', names=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        print(data[1])
        


        # Convert String values into integers
        le = preprocessing.LabelEncoder()
        x = le.fit_transform(data[8])

        data[8] = x
        print('------------------------------------------')
        print(data)
  
        # Check for messy data
        for i in range(data.shape[1] - 1):
            data = data[data[i] != '']
            data = data[data[i] != '?']
            data = data[data[i] != None]

        
        self.data = data[[1, 2, 3, 4, 5, 6, 7, 8]]
        self.targets = data[0]

    def split_data(self, size):
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.targets, test_size=size)
        return ( x_train, x_test, y_train, y_test )   

class Student:
    def __init__(self):
        self.data = []
        self.targets = []
    def load_data(self):
        # Read the file
        data = pd.read_csv('03_prove\data\student-mat.csv')
        print(data)

        # Convert String values into integers
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(data['target'])
        x_0 = le.fit_transform(data[0])
        x_1 = le.fit_transform(data[1])
        x_2 = le.fit_transform(data[2])
        x_3 = le.fit_transform(data[3])
        x_4 = le.fit_transform(data[4])
        x_5 = le.fit_transform(data[5])

        data['target'] = targets
        data[0] = x_0
        data[1] = x_1
        data[2] = x_2
        data[3] = x_3
        data[4] = x_4
        data[5] = x_5
  
        # Check for messy data
        for i in range(data.shape[1] - 1):
            data = data[data[i] != '']
            data = data[data[i] != '?']
            data = data[data[i] != None]

        self.data = data[[0, 1, 2, 3, 4, 5]]
        self.targets = data[['target']]

    def split_data(self, size):
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.targets, test_size=size)
        return ( x_train, x_test, y_train, y_test )

###################################################################
# Testing CAR
###################################################################
'''car = Car()
car.load_data()
car_x_train, car_x_test, car_y_train, car_y_test = car.split_data(.3)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(car_x_train, car_y_train.values.ravel())
predictions = classifier.predict(car_x_test)

print("\nAccuracy: " + str(classifier.score(car_x_test, car_y_test)))'''


###################################################################
# Testing MPG
###################################################################
'''mpg = Mpg()
mpg.load_data()
mpg_x_train, mpg_x_test, mpg_y_train, mpg_y_test = mpg.split_data(.3)

regr = KNeighborsRegressor(n_neighbors=3)
regr.fit(mpg_x_train, mpg_y_train)
#predictions = regr.predict(mpg_x_test)'''

###################################################################
# Testing STUDENT
###################################################################
student = Student()
student.load_data()
student_x_train, student_x_test, student_y_train, student_y_test = student.split_data(.3)

# regr = KNeighborsRegressor(n_neighbors=3)
# regr.fit(student_x_train, student_y_train)
# predictions = regr.predict(student_x_test)