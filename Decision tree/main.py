#############################################################################################
# Decision Tree (for iris data set)
# Author: Colten Larsen
#############################################################################################

import math

import numpy as np
import pandas as pd
import pydotplus
from IPython.display import Image
from sklearn import datasets, tree
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier, export_graphviz


class Node:
    def __init__(self, split=None, label=None, total_entropy=None):
        self.split = split
        self.label = label
        self.true_side = []
        self.false_side = []
        self.total_entropy = total_entropy


class Decision_Tree:
    def __init__(self):
        self.root = None

    ################################################################
    # entropy() takes in an array of values and calculates the entropy
    ################################################################
    def entropy(self, values):
        # Get percentages
        p = []
        uniques, counts = np.unique(values, return_counts=True)
        for i in range(len(np.unique(values))):
            p.append(counts[i] / len(values))

        # Calc Entropy
        entropy = 0
        for i in range(len(p)):
            if p[i] != 0:
                entropy += p[i] * math.log2(p[i])

        # Return Entropy
        if entropy == 0:
            return entropy
        return -entropy
    ################################################################
    # total_entropy() takes is an array of entropies (e) and an array
    # of percentages (p) and calculates the total_entropy
    ################################################################

    def total_entropy(self, e, p):
        # if the total is not equal to 1, throw and error
        if sum(p) != 1.0:
            print("---------------------------------\nERROR: Total does not equal 1\n---------------------------------")
            return None

        # return the total_entropy
        total_entropy = 0
        for i in range(len(e)):
            total_entropy += p[i] * e[i]

        return total_entropy

    ################################################################
    # equal_width_split() will take in an array of data and output
    # a number where the split will happen if split by the total
    # range of the data set
    ################################################################
    def equal_width_split(self, data):
        return (np.max(data) + np.min(data))/2

    ################################################################
    # total_entropy() takes in an array of data and output a number
    # where the split will happen if split if half by number
    # of values
    ################################################################
    def equal_frequency_split(self, data):
        return data[int(len(data)/2)]

    ################################################################
    # build_tree() will build the decision tree from data(the actual
    # data) targets(each target of each data points) and labels(the
    # name of the labels)
    ################################################################
    def build_tree(self, data, targets, labels):
        root = None

        # Check if all of the targets are equal
        if np.unique(targets).size == 1:
            return np.unique(targets)[0]

        # Check if there are any more labels
        elif len(labels) == 0:
            return (np.bincount(targets).argmax())

        # Find the best label and repeat build_tree
        else:
            # Calculate Split and create Nodes
            nodes = []
            j = 0
            for i in labels:
                nodes.append(Node(label=i, split=(
                    self.equal_width_split(np.sort(data[:, j])))))
                j = j + 1

            #nodes[0].true_side = [1,2,3]

            # Calculate total_entropy of each node
            for i in range(len(nodes)):
                for j in range(len(data)):
                    if data[j][i] >= nodes[i].split:
                        nodes[i].true_side.append((data[j], targets[j]))
                    else:
                        nodes[i].false_side.append((data[j], targets[j]))

                true_side_data, true_side_targets = zip(*nodes[i].true_side)
                false_side_data, false_side_targets = zip(*nodes[i].false_side)
                e = [(self.entropy(true_side_targets)),
                     (self.entropy(false_side_targets))]
                p = [((len(true_side_targets))/(len(data))),
                     ((len(false_side_targets))/(len(data)))]
                nodes[i].total_entropy = self.total_entropy(e, p)

            # Pick best Node
            best_entropy = min(node.total_entropy for node in nodes)
            for i in nodes:
                if i.total_entropy == best_entropy:
                    best_node = i

            # Set root
            root = best_node

            # Clean data and preform recursion
            # Right Side
            new_data, new_targets = zip(*best_node.true_side)
            new_data, new_targets = np.asarray(
                new_data), np.asarray(new_targets)
            new_labels = np.delete(labels, np.argwhere(best_node.label))
            best_node.true_side = self.build_tree(new_data, new_targets, new_labels)
            # Left Side
            new_data, new_targets = zip(*best_node.false_side)
            new_data, new_targets = np.asarray(
                new_data), np.asarray(new_targets)
            new_labels = np.delete(labels, np.argwhere(best_node.label))
            best_node.false_side = self.build_tree(new_data, new_targets, new_labels)
        
        # Set final root
        self.root = root


# Load the iris data set and create a Decision_tree object
iris = datasets.load_iris()
my_tree = Decision_Tree()
root = my_tree.build_tree(iris.data, iris.target, ([
                'sepal_length', 'sepal_height', 'petal_length', 'petal_height']))

iris = load_iris()
X = pd.DataFrame(iris.data[:, :], columns = iris.feature_names[:])
y = pd.DataFrame(iris.target, columns =["Species"])

# Defining and fitting a DecisionTreeClassifier instance
new_tree = DecisionTreeClassifier()
new_tree.fit(X,y)




'''When your assignment is complete, please answer the questions in this text file and upload it to I-Learn.


1. If your code is at GitHub, please provide a link to your classifier in that public repository. (If you uploaded your code to I-Learn, you may leave this question blank.)


2. Briefly describe your overall approach to the task and highlight the most difficult part of this assignment.
Overall there was a lot of work with drawing it all out on a white board and wrapping my head
around the ID3 algorithm.
THe most difficult part was organizing my data in a way that would work.

3. Describe the dataset that you used.
Iris is a staple data set that has 4 columns of numeric data

4. Describe your results on this dataset. (e.g., What was the size of the tree? How did your implementation compare to existing implementations? How did your decision tree compare to your kNN classifier)
The tree was only 4 layers deep including the root and the first true side was a leaf.

5. If applicable, please describe anything you did to go above and beyond and the results you saw.
As far as going above and beyond i created a second split function that split with frequency instead
of the range of the data. 

6. Please select the category you feel best describes your assignment:
1 - Some attempt was made
2 - Developing, but significantly deficient
3 - Slightly deficient, but still mostly adequate
4 - Meets requirements
5 - Shows creativity and excels above and beyond requirements

I would rate my work between 4-5.

7. Provide a brief justification (1-2 sentences) for selecting that category.
I took a lot of time to complete the assignment and learned a lot about trees. I did do some smaller 
steps to explore my tree with the second split function but ran out of time to really dive deep.

'''
