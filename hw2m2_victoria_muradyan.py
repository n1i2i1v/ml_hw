#imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, RepeatedKFold

#data maker
#makes the circles inside each other
X, y= make_circles(n_samples=1500, noise=0.3, factor = 0.2, random_state=69)
yn = []
for i in range(len(y)):
    if y[i] == 1:
        yn.append(1)
    else:
        yn.append(0)
tf = np.array(yn)
dataf = np.array(X)

#makes the globs
c1 = np.array(list(zip(np.random.normal(4.2,0.8,  size=750),
                      np.random.normal(4.2,0.69,  size=750))))
c0 = np.array(list(zip(np.random.normal(2.5,0.7,  size=750),
                      np.random.normal(2.5,0.7,  size=750))))
t = np.hstack((np.ones(750), np.zeros(750)))
data = np.vstack((c0, c1))
mask = np.arange(len(data))
np.random.shuffle(mask)
data = data[mask]
t = t[mask]
data_final = []
data_final = np.array(data_final)

target_final = []
target_final = np.array(target_final)

data_final = np.vstack((data, dataf))
target_final = np.hstack((t, tf))

plt.scatter(data_final[:,0], data_final[:,1], c=target_final)
plt.show();

mask = np.arange(len(data_final))
np.random.shuffle(mask)
data = data_final[mask]
y = target_final[mask]
data_train, y_train = data[:1500], y[:1500]


#model maker
#Random Forest Model Maker
model1 = RandomForestClassifier(n_estimators=2, max_depth=5)
model1.fit(data_train, y_train)

#SVC model maker
model2 = SVC(probability=True, degree=3, kernel="poly")
model2.fit(data_train, y_train)

#probablility to class
def proba_to_num(proba):
    p = []
    for x in proba:
        if x > 0.5: p.append(1)
        else: p.append(0)
    return np.array(p)




# Stacker Class
class Stacker:
    def __init__(self, models, blender, cv = 3):
        self.models = models
        self.blender = blender
        self.cv = cv
        self.stack_data = []
        self.y_test_final = []
        
    def train(self, data, y):
        
        kf = RepeatedKFold(n_splits=self.cv, n_repeats=1, random_state = 69) 

        for train_index, test_index in kf.split(data):
                x_train, x_test = data[train_index], data[test_index] 
                y_train, y_test = y[train_index], y[test_index]
                for index in range(len(self.models)):
                    self.models[index].fit(x_train, y_train)
                   
                # predict on test
                predictions = []
                for model in self.models:
                    predictions.append(model.predict_proba(x_test)[:, 1])
                predictions = np.vstack((np.array(p) for p in predictions)).T
                blending_data = np.hstack((predictions, x_test))
                self.stack_data.append(blending_data)
                self.y_test_final.append(y_test)
        
        #assign the new values 
        dat_shape = np.array(self.stack_data)
        x_arg = dat_shape.shape[0]*dat_shape.shape[1]
        stack_data_final = np.array(self.stack_data).reshape((x_arg, dat_shape.shape[2]))
        y_fin_test = np.array(self.y_test_final).ravel()

        # train blender
        self.blender.fit(stack_data_final, y_fin_test)
    
    def predict(self, data):
        predictions = [model.predict_proba(data)[:, 1]
                      for model in self.models]
        predictions = np.vstack((np.array(p) for p in predictions)).T
        blending_data = np.hstack((predictions, data))
        return self.blender.predict_proba(blending_data)        




stacker = Stacker([SVC(probability=True, degree=3, kernel="poly"),
                   RandomForestClassifier(n_estimators=2, max_depth=5)],
                  LogisticRegression())
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=69)

stacker.train(x_train, y_train)

pred = stacker.predict(x_test)

print("Random Forest Accuracy Score is")
print(accuracy_score(y[1200:], model1.predict(data[1200:])))
print("SVC Accuracy Score is")
print(accuracy_score(y[1200:], model2.predict(data[1200:])))
print("Stacker Accuracy Score is")
print(accuracy_score(y_test, proba_to_num(pred[:, 1])))