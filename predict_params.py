import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()
demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()
solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()
solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()
price_train = pd.read_csv('Price_Train.csv', header=None).as_matrix()

x_values = np.arange(-20, +20, 0.01)
cost_charging = np.zeros(x_values.shape)
cost_discharging = np.zeros(x_values.shape)

error_data = (demand_train[:, 10] - solar_train[:, 10]) - (demand_train_pred[:, 10] - solar_train_pred[:, 10])

(mu, sigma) = norm.fit(error_data)
for i in range(x_values.size):
	my_bid_quantity_charging = demand_train_pred[:, 10] + x_values[i]*sigma + 5
	my_bid_quantity_discharging = demand_train_pred[:, 10] + x_values[i]*sigma - 4
	
	# excess is positive CHARGING
	excess_or_deficit = my_bid_quantity_charging - demand_train[:, 10]
	excess_cost = (excess_or_deficit > 0).astype(np.int) * price_train[:, 10]
	deficit_cost = (excess_or_deficit < 0).astype(np.int) * np.ones(price_train[:,10].shape) * 7
	total_cost = np.sum(excess_cost) + np.sum(deficit_cost)
	cost_charging[i] = total_cost

	#DISCHARGING
	excess_or_deficit = my_bid_quantity_discharging - demand_train[:, 10]
	excess_cost = (excess_or_deficit > 0).astype(np.int) * price_train[:, 10]
	deficit_cost = (excess_or_deficit < 0).astype(np.int) * np.ones(price_train[:,10].shape) * 7
	total_cost = np.sum(excess_cost) + np.sum(deficit_cost)
	cost_discharging[i] = total_cost


print np.min(cost_charging), np.argmin(cost_charging), x_values[np.argmin(cost_charging)]
print np.min(cost_discharging), np.argmin(cost_discharging), x_values[np.argmin(cost_discharging)]
plt.plot(x_values, cost_charging)
plt.plot(x_values, cost_discharging)
plt.show()
