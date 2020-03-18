from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier

def fun():
    digits = datasets.load_digits()
    x_train_digits, x_test_digits, y_train_digits, y_test_digits = train_test_split(digits.data, digits.target, test_size=.3)

    wine = datasets.load_wine()
    x_train_wine, x_test_wine, y_train_wine, y_test_wine = train_test_split(wine.data, wine.target, test_size=.3)

    olivetti_faces = datasets.fetch_olivetti_faces()
    x_train_olivetti_faces, x_test_olivetti_faces, y_train_olivetti_faces, y_test_olivetti_faces = train_test_split(olivetti_faces.data, olivetti_faces.target, test_size=.3)

    classifier_digits = AdaBoostClassifier()
    classifier_wine = AdaBoostClassifier()
    classifier_olivetti_faces = AdaBoostClassifier()

    classifier_digits.fit(x_train_digits, y_train_digits)
    classifier_wine.fit(x_train_wine, y_train_wine)
    classifier_olivetti_faces.fit(x_train_olivetti_faces, y_train_olivetti_faces)

    targets_predicted_digits = classifier_digits.predict(x_test_digits)
    targets_predicted_wine = classifier_wine.predict(x_test_wine)
    targets_predicted_olivetti_faces = classifier_olivetti_faces.predict(x_test_olivetti_faces)

    j = 0
    for i in y_test_digits:
        if y_test_digits[i] == targets_predicted_digits[i]:
            j += 1
    digits_acc = "Accuracy of AdaBoost Digits:        " + str(100 * (j/len(y_test_digits)))
    j = 0
    for i in y_test_wine:
        if y_test_wine[i] == targets_predicted_wine[i]:
            j += 1
    wine_acc = "Accuracy of AdaBoost Wine:          " + str(100 * (j/len(y_test_wine)))
    j = 0
    for i in y_test_olivetti_faces:
        if y_test_olivetti_faces[i] == targets_predicted_olivetti_faces[i]:
            j += 1
    olivetti_faces_acc = "Accuracy of AdaBoost Olivetti Faces: " + str(100 * (j/len(y_test_olivetti_faces)))

    return (digits_acc, wine_acc, olivetti_faces_acc)