from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np


# read data from a text file. One number per line
solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()
solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()

mu_total = []
sigma_total = []

for i in range(12,13):
	datos = solar_train[:,i] - solar_train_pred[:,i]

	# best fit of data
	(mu, sigma) = norm.fit(datos)
	mu_total.append(mu)
	sigma_total.append(sigma)

	# the histogram of the data
	n, bins, patches = plt.hist(datos, 60, normed=1, facecolor='green', alpha=0.75)

	# add a 'best fit' line
	y = mlab.normpdf( bins, mu, sigma)
	l = plt.plot(bins, y, 'r--', linewidth=2)

	#plot
	plt.xlabel('Smarts')
	plt.ylabel('Probability')
	plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
	plt.grid(True)

	plt.show()
'''
x = open('data_obs_solar.txt', 'w')
for j in range(0,24):
	x.write("Mean =" + str(mu_total[j]) + " Std dev = " + str(sigma_total[j]) + " \n ")
x.close()
'''
