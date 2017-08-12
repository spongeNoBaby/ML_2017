from sklearn import datasets, linear_model
from sklearn import metrics
import numpy as np
import math

boston = datasets.load_boston()
x = boston.data[:, 12].reshape((506, 1))
y = boston.target

regr = linear_model.LinearRegression()
regr.fit(x, y)
print("Univariate Linear Regression:")
print("Coefficients：\n", regr.coef_)
print("Intercept: \n", regr.intercept_)
print("Residue sum of squares: %.2f" % np.mean((regr.predict(x) - y) ** 2))
print("R^2 score: %.2f" % regr.score(x, y))
print("\n")

## score = 1.0 is the best possible value, means y_predict = y_true
x = boston.data
regr.fit(x, y)
print("Multivariate Linear Regression:")
print("Coefficients: \n", regr.coef_)
print("Intercept: \n", regr.intercept_)
print("Residue sum of squares: %.2f" % np.mean((regr.predict(x) - y) ** 2))
print("R^2 score: %.2f" % regr.score(x, y))
print("\n")


train = np.random.choice([True, False], len(x), replace = True, p = [0.9, 0.1])
x_train = x[train, :]
y_train = y[train]

x_test = x[~train, :]
y_test = y[~train]

regr.fit(x_train, y_train)
print("R^2 Score: %.2f" % regr.score(x_test, y_test))

y_pred = regr.predict(x_test)
metrics.explained_variance_score(y_test, y_pred)
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)


x = boston.data
y = boston.target

mapped = []
for i in x:
    mapped.append([i[0]*i[0], i[5]*i[5], i[0]*i[5], i[0], i[5]])

mapped = np.asarray(mapped)

def map(orig_data, terms):
    mapped = []
    for x in orig_data:
        xx = []
        for d in terms:
            v = 1.0
            for pos, exponent in d.items():
                v *= math.pow(x[pos], exponent)
            xx.append(v)
        mapped.append(xx)
    return np.asarray(mapped)

terms = [{0:2}, {5:2}, {0:1, 5:1}, {0:1}, {5:1}, {}]

## mapped = x_mapped
x_mapped = map(x, terms)

regr = linear_model.LinearRegression()
regr.fit(mapped, y)
regr.score(mapped, y)
regr.intercept_
regr.coef_


regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(mapped, y)
regr.score(mapped, y)
regr.intercept_
regr.coef_


regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(x_mapped, y)
regr.score(x_mapped, y)
regr.intercept_
regr.coef_


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

diabetes = datasets.load_diabetes()
x = diabetes.data[:,[2,8]]
y = diabetes.target
regr = linear_model.LinearRegression()
regr.fit(x, y)
steps = 40
lx0 = np.arange(min(x[:,0]), max(x[:,0]), (max(x[:,0]) - min(x[:,0])) / steps).reshape(steps,1)
lx1 = np.arange(min(x[:,1]), max(x[:,1]), (max(x[:,1]) - min(x[:,1])) / steps).reshape(steps,1)
xx0, xx1 = np.meshgrid(lx0, lx1)
xx = np.zeros(shape = (steps,steps,2))

xx[:,:,0] = xx0
xx[:,:,1] = xx1
x_stack = xx.reshape(steps ** 2, 2)
y_stack = regr.predict(x_stack)
yy = y_stack.reshape(steps, steps)

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.scatter(x[:,0], x[:,1], y, color = 'red')
ax.plot_surface(xx0, xx1, yy, rstride=1, cstride=1)
plt.show()