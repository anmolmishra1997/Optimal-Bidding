import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()[:700, :]
demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()[:700, :]
solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()[:700, :]
solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()[:700, :]
price_train = pd.read_csv('Price_Train.csv', header=None).as_matrix()[:700, :]
price_train_pred = pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()[:700, :]

x_values = np.arange(-20, +20, 0.01)
cost = np.zeros(x_values.shape)

for hour in range(24):
	error_data = (price_train_pred[:, hour] - price_train[:, hour])

	(mu, sigma) = norm.fit(error_data)
	print sigma
	for i in range(x_values.size):
		my_bid_price = price_train_pred[:, hour] + x_values[i]*sigma
		result_bidding = (my_bid_price >= price_train[:, hour]).astype(np.int)
		total_cost = np.sum(result_bidding*price_train[:, hour]*demand_train[:, hour]) + np.sum((1 - result_bidding)*7.*demand_train[:, hour])
		cost[i] = total_cost 
	print "Hour {}".format(hour)
	print np.min(cost), np.argmin(cost), x_values[np.argmin(cost)]
	#print np.min(cost_discharging), np.argmin(cost_discharging), x_values[np.argmin(cost_discharging)]
	#plt.plot(x_values, cost)
	#plt.show()
