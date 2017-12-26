from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np


# read data from a text file. One number per line
demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()
demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()
solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()
solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()

plt.hist((demand_train[:, 10] - solar_train[:, 10]) - (demand_train_pred[:, 10] - solar_train_pred[:, 10]), bins=100)
plt.show()

plt.hist((demand_train[:, 10] - solar_train[:, 10])/(demand_train_pred[:, 10] - solar_train_pred[:, 10]), bins=100)
plt.show()

t = np.argsort((demand_train[:, 10] - solar_train[:, 10])/(demand_train_pred[:, 10] - solar_train_pred[:, 10]))
print t[0]
print demand_train[103, 10], demand_train_pred[103, 10], solar_train[103, 10], solar_train_pred[103, 10]

# Fitting a curve in 10th hour for the difference between solar and prediction


datos = (demand_train[:, 10] - solar_train[:, 10]) - (demand_train_pred[:, 10] - solar_train_pred[:, 10])

# best fit of data
(mu, sigma) = norm.fit(datos)

# the histogram of the data
n, bins, patches = plt.hist(datos, 60, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

#plot
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.grid(True)

plt.show()