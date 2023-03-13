# Author List:
# Marc Thurow
# Alexander Schmitz
# Artur Leinweber
# Mathias Bredereck

# TODO:
# - Visualisierung
# - Random Forest Algo, um das Overfitting zu verringern
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

epsilon = 10e-10
import matplotlib.pyplot as plt

def plotData(tiles, n, data, title, x_min, x_max, y_min, y_max):
    getColor = {-1:'red', 1:'blue'}

    x_pos = data["x"].values
    y_pos = data["y"].values
    color = data["Category"].apply(lambda value: getColor[value])

    tile = tiles[1][n]
    
    tile.set_aspect('equal')
    tile.set_title(title)
    tile.set_xlabel("x")
    tile.set_ylabel("y")
    tile.set_xlim(x_min, x_max)
    tile.set_ylim(y_min, y_max)
    tile.scatter(x = x_pos, y = y_pos, c = color)
class model:
    
    def __init__(self):
        pass
    
    def train(self, data, attributes, goalAttribute):
        pass
    
    def predict(self, featureVector, attributes, goalAttribute):
        pass
    
class DecisionTree(model):
    
    def __init__(self):
        super().__init__()
    
    def calculateEntropy(self, probabilities):
        
        entropy = 0
        for p in probabilities:
            entropy -= p*math.log(p,2)
        return entropy
    
    def calculateInformationGain(self, rootEntropy, childPropabilities, childEntropies):
        
        gain = rootEntropy
        for p, e in zip(childPropabilities, childEntropies):
            gain -= p*e
        return gain
    
    def calculateProbablity(data, attributes, goalAttribute):
        pass
    
    def splitDataOnAttribute(self, data, attributes, attribute):
        pass
    
    def getBestSplit(self, E_Root, attributes, data, goalAttribute):
        pass
    
    def createDecisionTree(self, data, attributes, goalAttribute):
        pass
class ContinuousDecisionTree(DecisionTree):
    
    def __init__(self):
        super().__init__()
        self.tree = None
        
    def _calculateEntropy(self, probabilities):
        
        entropy = super().calculateEntropy(probabilities)
        return entropy
    
    def _calculateInformationGain(self, rootEntropy, childPropabilities, childEntropies):
        
        gain = super().calculateInformationGain(rootEntropy, childPropabilities, childEntropies)
        return gain
    
    def _calculateProbablity(self, data, attributes):
        
        attributeIndex = len(attributes) -1
        attributeValues = {}
        
        for row in data:
            value = row[attributeIndex]
            if not value in attributeValues:
                attributeValues[value] = 1 / len(data)
            else:
                attributeValues[value] += (1 / len(data))
            
        return list(attributeValues.values())
    
    def splitDataOnAttribute(self, data, attributes, attribute):
        pass
    
    def splitOnAttributeValue(self, data, splitAfterIndex):
        
        data1 = data[0 : splitAfterIndex+1]
        data2 = data[splitAfterIndex+1 :]
        
        return data1, data2
    
    def _getBestSplit(self, E_Root, attributes, data):
        
        bestSplit = None
        decisionThreshold = None
        attributeIndex = None
        
        #find best split throughout all dims
        gain = 0
        for i, attribute in enumerate(attributes):
            
            if i == len(attributes) - 1:
                continue
            newData = sorted(data, key=lambda point : point[i])
            
            for j, row in enumerate(newData):
                
                if j == len(newData) -1:
                    continue
                
                splittedNewData = self.splitOnAttributeValue(newData, j)
                childEntropies = []
                childProbs = []

                for split in splittedNewData:
                    
                    probs = self._calculateProbablity(split, attributes)
                    childEntropy = self._calculateEntropy(probs)
                    childEntropies.append(childEntropy)
                    
                    childProbs.append(len(split) / len(newData))
                    
                newGain = self._calculateInformationGain(E_Root, childProbs, childEntropies)
                
                if newGain > gain:
                    
                    gain = newGain
                    bestSplit = splittedNewData
                    attributeIndex = i
                    decisionAxis = np.array(splittedNewData[0])[:,attributeIndex]
                    decisionThreshold = ( np.max(decisionAxis)  )
                    
        
        return bestSplit, decisionThreshold, attributeIndex
    
    def train(self, data, attributes, goalAttribute):
        self.tree = self.createDecisionTree(data, attributes, goalAttribute)
    
    def createDecisionTree(self, data, attributes, goalAttribute):
        
        decisionTree = {}
        
        probabilities = self._calculateProbablity(data, attributes)
        E_Root = self._calculateEntropy(probabilities)
        
        if np.abs(E_Root) > 0 + epsilon:
        
            bestSplit, decisionThreshold, attributeIndex = self._getBestSplit(E_Root, attributes, data)
            
            subtrees = []
            for dataParts in bestSplit:
                
                newData = np.array(dataParts[:])
                
                subtree = self.createDecisionTree(newData, attributes, goalAttribute)
                subtrees.append(subtree)
                
            decisionTree[attributes[attributeIndex]] = [decisionThreshold, subtrees]
            
        else:
            leaf = {goalAttribute : data[0][-1]}
            decisionTree = leaf
            
        return decisionTree
    
    def predict(self, featureVector, attributes, goalAttribute):
        
        classPredict = None
        leafFound = False
        
        currentTreeView = self.tree
        
        while not leafFound:
            
            currentNodeAttribute = list(currentTreeView.keys())[0]
            
            verticesNodes = list(currentTreeView.values())
            
            if currentNodeAttribute == goalAttribute: #leaf found
                classPredict = verticesNodes[0] #just the class to find
                leafFound = True

            if not leafFound:
            
                attributeIndex = attributes.index(currentNodeAttribute)
                featureAttribute = featureVector[attributeIndex]
                
                childNodeIndex = 0
                if featureAttribute <= verticesNodes[0][0]:
                    childNodeIndex = 0
                else:
                    childNodeIndex = 1
                
                #now go for the child node
                currentTreeView = verticesNodes[0][1][childNodeIndex]
        
        return classPredict
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() 
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

dataC = read_data(r"../input/kiwhs-comp-1-complete/train.arff")
import numpy as np
import pandas as pd

data = pd.DataFrame({'x':[item[0] for item in dataC], 'y':[item[1] for item in dataC], 'Category':[item[2] for item in dataC]})
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, random_state = 0, test_size = 0.2)
tiles = plt.subplots(1, 3, figsize=(30, 10))

plotData(tiles, 0, data, "data", -5, 5, -5, 5)
plotData(tiles, 1, train_data, "train data", -5, 5, -5, 5)
plotData(tiles, 2, test_data, "test data", -5, 5, -5, 5)
goalAttribute = 'class'
attributesC = ["x","y","class"]
cdt = ContinuousDecisionTree()
cdt.train(train_data.values, attributesC, goalAttribute)

print(cdt.tree)
accuracy = 0
prediction_data = pd.DataFrame(columns = ['x', 'y', 'Category'])

for index, data in test_data.iterrows():
    prediction = cdt.predict([data['x'], data['y']], attributesC, 'class')
    category = data['Category']
    prediction_data = prediction_data.append(pd.Series({'x' : data['x'], 'y' : data['y'], 'Category' : prediction}), ignore_index=True)
    if prediction == category:
        accuracy += 1
accuracy /= len(test_data)    

print(accuracy)
tiles = plt.subplots(1, 2, figsize=(20, 10))

plotData(tiles, 0, test_data, "test data", -5, 5, -5, 5)
plotData(tiles, 1, prediction_data, "prediction data", -5, 5, -5, 5)
#Plotting test-data classification and boundary
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_decision_boundary(model,X,y):
    h = .02  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    xx, yy = np.meshgrid(np.arange(np.min(X) - 1, np.max(X) + 1, h),
                             np.arange(np.min(y) - 1, np.max(y) + 1, h))

    points = np.c_[xx.ravel(),yy.ravel()]

    Z = np.zeros(len(points))
    for i, point in enumerate(points):
        prediction = model.predict(point, attributesC, 'class')
        Z[i] = prediction

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(5,5))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
X = np.array(train_data.values)[:,0]
y = np.array(train_data.values)[:,1]

plot_decision_boundary(cdt, X, y)

plt.scatter(np.array(train_data.values)[:,0], np.array(train_data.values)[:,1], c=np.array(train_data.values)[:,2], cmap=cmap_bold, edgecolor='k', s=20)

plt.show()
class Forest(model):
    
    def __init__(self, noTrees=2):
        super().__init__()
        
        self.trees = []
        
        for i in range(noTrees):
            self.trees.append(ContinuousDecisionTree())
            
        self.accuracies = np.ones(len(self.trees))
    
    def train(self, data, attributes, goalAttribute):
        
        train_data_values = data[:]
        data_fields = []
        data_field_size = int(len(train_data_values)-0.5*len(train_data_values))
        
        
        for i in range(len(self.trees)):
            
            np.random.shuffle(train_data_values)
            data_fields.append(train_data_values[0:data_field_size])
        
        print(data_field_size, np.array(data_fields).shape)
        #old:
        #np.array_split(train_data_values, len(self.trees))
        
        for tree, data_field in zip(self.trees, data_fields):
            tree.train(data_field, attributes, goalAttribute)
    
    def predict(self, featureVector, attributes, goalAttribute):
        
        forest_prediction = {}
        for index, tree in enumerate(self.trees):

            this_tree_prediction = tree.predict(featureVector, attributes, goalAttribute)
            if not this_tree_prediction in forest_prediction:
                forest_prediction[this_tree_prediction] = self.accuracies[index]
            else:
                forest_prediction[this_tree_prediction] += self.accuracies[index]
        
        prediction = sorted(forest_prediction.items(), key=lambda kv : kv[1], reverse=True)[0][0]
        
        return prediction
    
    def defineTreeWeights(self, probabilities):
        for i in range(len(self.accuracies)):
            self.accuracies[i] = probabilities[i]
forest = Forest(10)
goalAttribute = 'class'
attributesC = ["x","y","class"]

forest.train(list(train_data.values), attributesC, goalAttribute)
X = np.array(train_data.values)[:,0]
y = np.array(train_data.values)[:,1]

plot_decision_boundary(forest, X, y)

plt.scatter(np.array(train_data.values)[:,0], np.array(train_data.values)[:,1], c=np.array(train_data.values)[:,2], cmap=cmap_bold, edgecolor='k', s=20)

plt.show()
accuracies = np.zeros(len(forest.trees))
accuracy = 0
prediction_data = pd.DataFrame(columns = ['x', 'y', 'Category'])

for index, data in test_data.iterrows():
    
    category = data['Category']
    predictionForest = forest.predict([data['x'], data['y']], attributesC, 'class')
    prediction_data = prediction_data.append(pd.Series({'x' : data['x'], 'y' : data['y'], 'Category' : predictionForest}), ignore_index=True)
    if predictionForest == category:
        accuracy += 1
    
    for i, tree in enumerate(forest.trees):
        
        predictiontree = tree.predict([data['x'], data['y']], attributesC, 'class')
        prediction_data = prediction_data.append(pd.Series({'x' : data['x'], 'y' : data['y'], 'Category' : predictiontree}), ignore_index=True)
        
        if predictiontree == category:
            accuracies[i] += 1

accuracy /= len(test_data)
print(accuracy)
print()
print(accuracies / len(test_data))
forest.defineTreeWeights(accuracies/len(test_data))
accuracy = 0
prediction_data = pd.DataFrame(columns = ['x', 'y', 'Category'])

for index, data in test_data.iterrows():
    
    category = data['Category']
    predictionForest = forest.predict([data['x'], data['y']], attributesC, 'class')
    prediction_data = prediction_data.append(pd.Series({'x' : data['x'], 'y' : data['y'], 'Category' : predictionForest}), ignore_index=True)
    if predictionForest == category:
        accuracy += 1

accuracy /= len(test_data)
print(accuracy)
tiles = plt.subplots(1, 2, figsize=(20, 10))

plotData(tiles, 0, test_data, "test data", -5, 5, -5, 5)
plotData(tiles, 1, prediction_data, "prediction data", -5, 5, -5, 5)
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() 
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

dataC = read_data(r"../input/kiwhs-comp-1-complete/train.arff")
train_data = pd.DataFrame({'x':[item[0] for item in dataC], 'y':[item[1] for item in dataC], 'Category':[item[2] for item in dataC]})
goalAttribute = 'class'
attributesC = ["x","y","class"]

cdt = ContinuousDecisionTree()
cdt.train(train_data.values, attributesC, goalAttribute)

print("well done")
test_data = pd.read_csv(r"../input/kiwhs-comp-1-complete/test.csv")
test_prediction = pd.DataFrame(columns = ['Id (String)', 'Category (String)'])

for index, data in test_data.iterrows():
    prediction = cdt.predict([data['X'], data['Y']], attributesC, 'class')
    test_prediction = test_prediction.append(pd.Series({'Id (String)' : int(data['Id']), 'Category (String)' : int(prediction)}), ignore_index=True)
    
test_prediction.to_csv(r"predict.csv", index=False)