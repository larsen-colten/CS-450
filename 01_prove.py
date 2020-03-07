from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.3)


classifier = GaussianNB()
classifier.fit(X_train, y_train)

targets_predicted = classifier.predict(X_test)

j = 0
for i in y_test:
    if y_test[i] == targets_predicted[i]:
        j += 1

print("Accuracy of Gaussian: " + str(100 * (j/len(y_test))))

class HardCodedClassifier:
    def fit(self, x_train, y_train):
        return None
        
    def predict(self, data):
        j = []
        for i in data:
            j.append(0)

        return j

classifier = HardCodedClassifier()
classifier.fit(X_train, y_train)

targets_predicted = classifier.predict(X_test)

j = 0
for i in y_test:
    if y_test[i] == targets_predicted[i]:
        j += 1

print("Accuracy of Hard Coded: " + str(100 * (j/len(y_test))))