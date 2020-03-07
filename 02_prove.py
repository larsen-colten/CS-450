from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np




iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.3)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

print("\nAccuracy: " + str(classifier.score(x_test, y_test)))

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
df = sns.load_dataset('iris')
sns.pairplot(df, hue='species', height=2.5)
plt.show()

'''
When your assignment is complete, please answer the questions in this text file and upload it to I-Learn.

1. If you did not include your source code in your I-Learn submission, please provide the URL of your public GitHub repository.


2. Briefly describe your overall approach to the task and highlight the most difficult part of this assignment.
Reading through the documentation of scikitlearn.

3. Describe your results for the Iris data set. (For example, what level of accuracy did you see for different values of K?)
The accuracy varies from around 100 - 95 percent. Sometime it would drop to 60% but hardly at all.

4. How did your implementation compare to the existing implementation?
This was very simple adn easy to use. It was most likely faster and more efficient.

5. Describe anything you did to go above and beyond the minimum standard requirements.
I added a graph implementation using seaborn. I first i was using matplotlib but was recommended seaborn by
Brother Burton.

6. Please select the category you feel best describes your assignment:
1 - Some attempt was made
2 - Developing, but significantly deficient
3 - Slightly deficient, but still mostly adequate
4 - Meets requirements
5 - Shows creativity and excels above and beyond requirements
5

7. Provide a brief justification (1-2 sentences) for selecting that category.
I feel that I should get a 5 on this assignment because it completed the assignment and I went
above and beyond with the graph implimentation. I did try multiple graphing libraries and landed on seaborn.
'''
