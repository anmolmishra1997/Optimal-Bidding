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

for block in range(int(demand_train.shape[0]/900)):
	actual_demand_train_block = actual_demand_train[block*900:(block+1)*900, :].ravel()
	actual_demand_train_pred_block = actual_demand_train_pred[block*900:(block+1)*900, :].ravel()
	price_train_block = price_train[block*900:(block+1)*900, :].ravel()
	price_train_pred_block = price_train_pred[block*900:(block+1)*900, :].ravel()

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
		(mu, sigma_price_) = norm.fit(error_price)

		sigma_quantity.append(sigma_quantity_)
		sigma_price.append(sigma_price_)

	overall_sigma_quantity.append(sigma_quantity)
	overall_sigma_price.append(sigma_price)

sigma_quantity = np.mean(np.asarray(overall_sigma_quantity), axis=0)
sigma_price = np.mean(np.asarray(overall_sigma_price), axis=0)

# Calculate on every 50 days for optimal x and average it

x_charge = []
x_discharge = []
x_neutral = []
y_price = []

for block in range(int(demand_train.shape[0]/900)):
	actual_demand_train_block = actual_demand_train[block*900:(block+1)*900, :].ravel()
	actual_demand_train_pred_block = actual_demand_train_pred[block*900:(block+1)*900, :].ravel()
	price_train_block = price_train[block*900:(block+1)*900, :].ravel()
	price_train_pred_block = price_train_pred[block*900:(block+1)*900, :].ravel()

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

	x_values = np.arange(-20, +20, 0.1)
	y_values = np.arange(-20, +20, 0.1)

	cost_charging = np.zeros((x_values.size, y_values.size))
	cost_discharging = np.zeros((x_values.size, y_values.size))
	cost_neutral = np.zeros((x_values.size, y_values.size))

	x_values_charge = []
	x_values_discharge = []
	x_values_neutral = []

	y_values_price = []

	for k in range(10):
		for i in range(x_values.size):
			for j in range(y_values.size):
				bid_qty_charge = (demand_pred_list[k] + x_values[i]*sigma_quantity[k] + 5).clip(min=0)
				bid_qty_discharge = (demand_pred_list[k] + x_values[i]*sigma_quantity[k] - 4).clip(min=0)
				bid_qty_neutral = (demand_pred_list[k] + x_values[i]*sigma_quantity[k]).clip(min=0)

				bid_price = (price_pred_list[k] + y_values[j] * sigma_price[k]).clip(max=7)

				result_bid = (bid_price >= price_list[k]).astype(np.int)

				#CHARGE
				cost_if_won = result_bid * (bid_qty_charge * bid_price + (demand_list[k] - bid_qty_charge).clip(min=0)*7)
				cost_if_lost = (1 - result_bid) * ((7*demand_list[k]).clip(min=0))
				cost_charging[i][j] = np.sum(cost_if_won) + np.sum(cost_if_lost)

				#DISCHARGE
				cost_if_won = result_bid * (bid_qty_discharge * bid_price + ((demand_list[k] - bid_qty_discharge).clip(min=0))*7)
				cost_if_lost = (1 - result_bid) * ((7*demand_list[k]).clip(min=0))
				cost_discharging[i][j] = np.sum(cost_if_won) + np.sum(cost_if_lost)

				#NEUTRAL
				cost_if_won = result_bid * (bid_qty_neutral * bid_price + ((demand_list[k] - bid_qty_neutral).clip(min=0))*7)
				cost_if_lost = (1 - result_bid) * ((7*demand_list[k]).clip(min=0))
				cost_neutral[i][j] = np.sum(cost_if_won) + np.sum(cost_if_lost)

		x_values_charge.append(x_values[np.unravel_index(np.argmin(cost_charging), cost_charging.shape)[0]])
		x_values_discharge.append(x_values[np.unravel_index(np.argmin(cost_discharging), cost_discharging.shape)[0]])
		x_values_neutral.append(x_values[np.unravel_index(np.argmin(cost_neutral), cost_neutral.shape)[0]])

		y_values_price.append(y_values[np.unravel_index(np.argmin(cost_charging), cost_charging.shape)[1]])

	x_charge.append(x_values_charge)
	x_discharge.append(x_values_discharge)
	x_neutral.append(x_values_neutral)

	y_price.append(y_values_price)

x_charge = np.mean(np.asarray(x_charge), axis=0)
x_discharge = np.mean(np.asarray(x_discharge), axis=0)
x_neutral = np.mean(np.asarray(x_neutral), axis=0)

y_price = np.mean(np.asarray(y_price), axis=0)



final = np.vstack((sigma_quantity, sigma_price, x_charge, x_discharge, x_neutral, y_price))

np.savetxt('blocks_900.txt', final, fmt='%.3e')