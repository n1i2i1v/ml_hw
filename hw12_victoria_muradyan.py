from collections import defaultdict, Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
class DecisionNode:
    def __init__(self, index=None, 
                 value=None, 
                 left_node=None, 
                 right_node=None, 
                 current=None, 
                 is_leaf=False):
        self.index = index
        self.value = value
        self.left_node = left_node
        self.right_node = right_node
        self.current = current
        self.is_leaf = is_leaf

class DecisionTree(object):

    def __init__(self, loss, max_depth):
        if loss == "entropy":
            self.loss = self.entropy
        else:
            self.loss = self.gini
        self.max_depth = max_depth
        self.tree = None
                
    def entropy(self, data1, data2):
    
        param1, param2 = self.paramify(data1), self.paramify(data2)
        entropy1 = -len(data) * sum((val / len(data1)) * np.log2(val / len(data1)) for feat, val in param1.items())
        entropy2 = -len(data) * sum((val / len(data2)) * np.log2(val / len(data2)) for feat, val in param2.items())

        return entropy1 + entropy2
        
    def gini(self, data1, data2):
    
        param1, param2 = self.paramify(data1), self.paramify(data2)
        gini1 = len(data) * sum((val / len(data1)) * (1 - val / len(data1)) for feat, val in param1.items())
        gini2 = len(data) * sum((val / len(data2)) * (1 - val / len(data2)) for feat, val in param2.items())

        return gini1 + gini2

    def train(self, data, y):
        y = np.reshape(y, (np.size(y), 1))
        data_all = np.concatenate((data, y), axis=1)
        self.tree = self.treeify(data_all, max_depth=self.max_depth)
        
    
    def treeify(self, data, current_depth=0, max_depth=1e8):

        if len(data) == 0:
            return DecisionNode(is_leaf=True)

        if current_depth == max_depth or len(self.paramify(data)) == 1:
            return DecisionNode(current = self.paramify(data), is_leaf=True)

        self_loss = self.loss(data, [])
        best_loss = 10000
        best_index = None
        best_value = None
        best_data = None

        features = np.array([[row[i] for row in data] for i in range(len(data[0]) - 1)])

        for index, feature in enumerate(features):
            feature = np.unique(feature)
            feature = [np.percentile(feature, i) for i in np.arange(0, 101, 25)]

            for value in feature:
                data1, data2 = self.spliter(data, index, value)
                loss = self.loss(data1, data2)
                if loss < best_loss:
                    best_loss = loss
                    best_index = index
                    best_value = value
                    best_data = data1, data2

        if abs(self_loss - best_loss) < 1e-10:
            return DecisionNode(current = self.paramify(data), is_leaf=True)
        else:
            return DecisionNode(index=best_index, value=best_value,
                                left_node= self.treeify(best_data[1], current_depth + 1, max_depth),
                                right_node= self.treeify(best_data[0], current_depth + 1, max_depth),
                                current = self.paramify(data))
    
    def paramify(self, data):
        results = defaultdict(int)
        for datum in data:
            r_s = datum[len(datum) - 1]
            results[r_s] += 1
        return dict(results)
    
    def spliter(self, data, f_index, f_value):
    
        data1, data2 = [], []
        for datum in data:
            if datum[f_index] >= f_value:
                data1.append(datum)
            else:
                data2.append(datum)
        return data1, data2
    
    def score(self, real, predicted):
        real = real.flatten()
        predicted = predicted.astype(float).flatten()
        sum_all = 0
        for i in range(len(predicted)):
            if predicted[i] == real[i]:
                sum_all += 1
        return 100 * sum_all / len(predicted)
        
    def predict_s(self, tree, main):
        if len(tree.current) == 1 or tree.is_leaf:
            return max(tree.current, key=tree.current.get)
        else:
            if main[tree.index] >= tree.value:
                return self.predict_s(tree.right_node, main)
            else:
                return self.predict_s(tree.left_node, main)

    def predict(self, data):
        return np.array([np.array([self.predict_s(self.tree, datum)]) for datum in data])
class RandomForest(object):
    def __init__(self, amount, max_depth, ratio):

        self.amount= amount
        self.max_depth = max_depth
        self.ratio = ratio
        self.tree_all = None

    def train(self, data, y):
        tree_all = []
        size = np.int(self.ratio * len(x_train))
        for each in range(self.amount):
            amount_n = data.shape[0]
            shuf = np.arange(amount_n)
            np.random.seed(each)
            np.random.shuffle(shuf)
            data, y = data[shuf], y[shuf]
            data, y = data[:size, :], y[:size]
            data, y = np.array(data), np.array(y)
            tree = DecisionTree("entropy", self.max_depth)
            tree.train(data, y)
            tree_all.append(tree)
        self.tree_all = tree_all

    def predict(self, data):
        targets = []

        def predict(current, datum):
            if current.is_leaf:
                key = max(current.current, key=current.current.get)
                return key
            if datum[current.index] >= current.value:
                current = current.right_node
                return predict(current, datum)
            else:
                current = current.left_node
                return predict(current, datum)

        for data in data:
            predict_all = []
            for tree_ind in self.tree_all:
                predict_all.append(predict(tree_ind.tree, data))
            predict_fin = np.array(predict_all)
            count = Counter(predict_fin)
            targets.append(count.most_common(1)[0][0])

        return np.array(targets)
model_tree = DecisionTree("entropy", 5000)
iris = load_iris()
data = iris.data
target = iris.target
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=69)
model_tree.train(x_train, y_train)
pred = model_tree.predict(x_test)
print("Decision Tree Score")
print(model_tree.score(pred, y_test))
model_forest = RandomForest(69, 420, 0.69)
model_forest.train(x_train, y_train)
pred_for = model_forest.predict(x_test)
print("Random Forest Score")
print(model_tree.score(pred_for, y_test))