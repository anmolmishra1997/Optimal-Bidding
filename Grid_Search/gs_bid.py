import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()[:100, :]
demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()[:100, :]
solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()[:100, :]
solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()[:100, :]
price_train = pd.read_csv('Price_Train.csv', header=None).as_matrix()[:100, :]
price_train_pred = pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()[:100, :]

x_values = np.arange(-10, +10, 0.1)
y_values = np.arange(-10, +10, 0.1)
cost_charging = np.zeros((x_values.size, y_values.size))
cost_discharging = np.zeros((x_values.size, y_values.size))
cost_neutral = np.zeros((x_values.size, y_values.size))

for hour in range(24):

	error_data_quantity = (demand_train[:, hour] - solar_train[:, hour]) - (demand_train_pred[:, hour] - solar_train_pred[:, hour])
	error_data_price = (price_train_pred[:, hour] - price_train[:, hour])

	(mu_error_quantity, sigma_error_quantity) = norm.fit(error_data_quantity)
	(mu_error_price, sigma_error_price) = norm.fit(error_data_price)
	print sigma_error_price, sigma_error_quantity

	for i in range(x_values.size):
		for j in range(y_values.size):
			my_bid_quantity_charging = (demand_train_pred[:, hour] - solar_train_pred[:, hour] + x_values[i]*sigma_error_quantity + 5).clip(min=0)
			my_bid_quantity_discharging = (demand_train_pred[:, hour] - solar_train_pred[:, hour] + x_values[i]*sigma_error_quantity - 4).clip(min=0)
			my_bid_quantity_neutral = (demand_train_pred[:, hour] - solar_train_pred[:, hour] + x_values[i]*sigma_error_quantity).clip(min=0)

			my_bid_price = price_train_pred[:, hour] + y_values[j]*sigma_error_price
			result_bidding = (my_bid_price >= price_train[:, hour]).astype(np.int)
			
			# excess is positive CHARGING
			cost_if_won = result_bidding * (my_bid_quantity_charging * my_bid_price  + (demand_train[:, hour] - my_bid_quantity_charging - solar_train[:, hour]).clip(min=0) * 7)
			cost_if_lost = (1 - result_bidding) * (7 * (demand_train[:, hour] - solar_train[:, hour]))
			total_cost = np.sum(cost_if_won) + np.sum(cost_if_lost)
			cost_charging[i, j] = total_cost

			#DISCHARGING
			cost_if_won = result_bidding * (my_bid_quantity_discharging * my_bid_price  + (demand_train[:, hour] - my_bid_quantity_discharging - solar_train[:, hour]).clip(min=0) * 7)
			cost_if_lost = (1 - result_bidding) * (7 * demand_train[:, hour] - solar_train[:, hour])
			total_cost = np.sum(cost_if_won) + np.sum(cost_if_lost)
			cost_discharging[i, j] = total_cost

			#NEUTRAL
			cost_if_won = result_bidding * (my_bid_quantity_neutral * my_bid_price  + (demand_train[:, hour] - my_bid_quantity_neutral - solar_train[:, hour]).clip(min=0) * 7)
			cost_if_lost = (1 - result_bidding) * (7 * demand_train[:, hour] - solar_train[:, hour])
			total_cost = np.sum(cost_if_won) + np.sum(cost_if_lost)
			cost_neutral[i, j] = total_cost

	print "Hour {}".format(hour)
	print np.min(cost_charging), np.unravel_index(np.argmin(cost_charging), cost_charging.shape), x_values[np.unravel_index(np.argmin(cost_charging), cost_charging.shape)[0]], y_values[np.unravel_index(np.argmin(cost_charging), cost_charging.shape)[1]]
	print np.min(cost_discharging), np.unravel_index(np.argmin(cost_discharging), cost_discharging.shape), x_values[np.unravel_index(np.argmin(cost_discharging), cost_discharging.shape)[0]], y_values[np.unravel_index(np.argmin(cost_discharging), cost_discharging.shape)[1]]
	print np.min(cost_neutral), np.unravel_index(np.argmin(cost_neutral), cost_neutral.shape), x_values[np.unravel_index(np.argmin(cost_neutral), cost_neutral.shape)[0]], y_values[np.unravel_index(np.argmin(cost_neutral), cost_neutral.shape)[1]]
