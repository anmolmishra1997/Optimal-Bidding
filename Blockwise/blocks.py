import numpy as np
import pandas as pd
from scipy.stats import norm


demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()
demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()
solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()
solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()
price_train = pd.read_csv('Price_Train.csv', header=None).as_matrix()
price_train_pred = pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()

actual_demand_train = demand_train - solar_train
actual_demand_train_pred = demand_train_pred - solar_train_pred

overall_sigma_quantity = []
overall_sigma_price = []

for block in range(int(demand_train.shape[0]/50)):
	actual_demand_train_block = actual_demand_train[block*50:(block+1)*50, :].ravel()
	actual_demand_train_pred_block = actual_demand_train_pred[block*50:(block+1)*50, :].ravel()
	price_train_block = price_train[block*50:(block+1)*50, :].ravel()
	price_train_pred_block = price_train_pred[block*50:(block+1)*50, :].ravel()

	sorted_demand_train_pred_block = np.sort(actual_demand_train_pred_block)
	sorted_demand_train_block = actual_demand_train_block[np.argsort(actual_demand_train_pred_block)]
	sorted_price_train_block = price_train_block[np.argsort(actual_demand_train_pred_block)]
	sorted_price_train_pred_block = price_train_pred_block[np.argsort(actual_demand_train_pred_block)]

	slice_values = np.array([15, 20, 25, 40, 60, 75, 90, 110, 160])

	copy_demand_pred = sorted_demand_train_pred_block
	copy_demand = sorted_demand_train_block
	copy_price_pred = sorted_price_train_pred_block
	copy_price = sorted_price_train_block

	demand_pred_list = np.split(copy_demand_pred, np.searchsorted(copy_demand_pred, slice_values, side='right'))
	demand_list = np.split(copy_demand, np.searchsorted(copy_demand_pred, slice_values, side='right'))
	price_pred_list = np.split(copy_price_pred, np.searchsorted(copy_demand_pred, slice_values, side='right'))
	price_list = np.split(copy_price, np.searchsorted(copy_demand_pred, slice_values, side='right'))

	sigma_quantity = []
	sigma_price = []

	for j in range(10):
		# MEAN TO BE ADJUSTED

		error_quantity = demand_list[j] - demand_pred_list[j]
		error_price = price_list[j] - price_pred_list[j]

		(mu, sigma_quantity_) = norm.fit(error_quantity)
		print j, "block"
		print "Q", sigma_quantity_
		(mu, sigma_price_) = norm.fit(error_price)
		print "P", sigma_price_

		sigma_quantity.append(sigma_quantity_)
		sigma_price.append(sigma_price_)

	overall_sigma_quantity.append(sigma_quantity)
	overall_sigma_price.append(sigma_price)














	

