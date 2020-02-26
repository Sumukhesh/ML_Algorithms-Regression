import pandas as pd 
import numpy as np 
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle 
from matplotlib import style

style.use("ggplot")
data = pd.read_csv("student-mat2.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "school", "famrel"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

#TO TRAIN THE MODEL MULTIPLE TIMES
best = 0 
for _ in range(60):
	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

	linear = linear_model.LinearRegression()
	linear.fit(x_train,y_train)
	acc = linear.score(x_test,y_test)
	print(acc)
	if best < acc:
		best = acc
		with open("studentmodel.pickle", "wb") as f:
			pickle.dump(linear, f)
#TO LOAD THE MODEL
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("-----------------------------")
print('coefficient: \n', linear.coef_)
print('Intercept:\n', linear.intercept_)
print("-----------------------------")


predictions = linear.predict(x_test)
for x in range(len(predictions)):
	print(predictions[x], x_test[x], y_test[x])

print(acc)


#DRAWING A MODEL ON THE GRAPH
p = "school"
plt.scatter(data[p],data[predict])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
