import numpy as np
import math

x = np.array([3, 6])
y = np.array([5, 2])
data = np.array([[2, 3], [3, 4], [5, 7], [2, 7], [3, 2], [1, 2], [9, 3], [4, 1], [1, 1], [5, 1]])
animals = ["dog", "cat", "bird", "fish", "fish", "dog", "cat", "dog", "cat", "fish"]
animal_index = [0, 1, 2, 3, 3, 0, 1, 0]
animal_classes = ['dog', 'cat', 'bird', 'fish']

def eucl_dist(x, y):
    return math.sqrt(pow(int(x[0] - y[0]), 2) + pow(int(x[1] - y[1]), 2))

def knn(k, data, labels, input):
    # Find distances
    dist = np.shape(data)[0]
    dist = np.zeros(dist)

    for i in range(data.shape[0]):
        dist[i] = eucl_dist(input, data[i])

    # Print Distances
    print("\nDistances: " +
    str(dist) + '\n-------------------------------------------')

    # Sort Distances
    sort_dist = np.argsort(dist)
    print("Sorted Distance indexs: " +
    str(sort_dist) + '\n-------------------------------------------')

    # Grab classes
    classes = []
    for i in range(k):
        classes = np.append(classes, animal_index[sort_dist[i]])

    print("K classes: " +
    str(classes) + '\n-------------------------------------------')

    # Average the k values
    avg_k = np.bincount(classes.astype(int)).argmax()

    print("Average k: " +
    str(avg_k) + '\n-------------------------------------------')

    # Output the closest animal class
    return(animal_classes[avg_k])

print("\nDistance between x and y: " + str(math.sqrt(pow(int(x[0] - y[0]), 2) + pow(int(x[1] - y[1]), 2))) + 
'\n-------------------------------------------')
print("Distance between x and y using function: " + str(eucl_dist(x, y)) + 
'\n-------------------------------------------')

input = np.array([3,2])
k = 4
output = knn(k, data, animals, input)

print("Output: " + output)

########################################################################
# Mat plot lib
import matplotlib.pyplot as plt
# Split data
graph_x, graph_y = [], []
for i in data:
    graph_x = np.append(graph_x, i[0])
    graph_y = np.append(graph_y, i[1])

# Plot ( individual class)
#for i in range(data.shape[0]):
#    if animals[i] == 'dog':
#        plt.scatter(graph_y[i], graph_x[i], label="DOG")
#    if animals[i] == 'cat':
#       plt.scatter(graph_y[i], graph_x[i], label="CAT")
#    if animals[i] == 'bird':
#        plt.scatter(graph_y[i], graph_x[i], label="BIRD")
#    if animals[i] == 'fish':
#       plt.scatter(graph_y[i], graph_x[i], label="FISH")

# Plot ( regular )
#plt.scatter(graph_x, graph_y, label='Data')
#plt.scatter(input[0], input[1], label='Input')
#plt.legend(numpoints=1)
#plt.show()

#seaborn
import seaborn as sns

sns.set()
df = sns.load_dataset('iris')
sns.pairplot(df, hue='species', height=2.5)

plt.show()