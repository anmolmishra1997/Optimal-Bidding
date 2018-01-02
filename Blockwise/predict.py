import numpy as np
from scipy_optimize import charge_discharge
import pandas as pd


std_price = []
x_values_price = []
std_demand_quantity = []
x_values_charging = []
x_values_discharging = []
x_values_neutral = []


#READING FROM FILE
with open('blocks_2.txt') as f:
	content = f.readlines()

var = [std_demand_quantity, std_price, x_values_charging, x_values_discharging, x_values_neutral, x_values_price]
for i in range(len(content)):
	for j in range(10):
		var[i].append(float(content[i].split()[j]))


demand_test_pred = (pd.read_csv('Demand_LB_pred.csv', header=None).as_matrix()).ravel()
solar_test_pred = (pd.read_csv('Solar_LB_pred.csv', header=None).as_matrix()).ravel()
price_test_pred = (pd.read_csv('Price_LB_pred.csv', header=None).as_matrix()).ravel()

actual_test_pred = demand_test_pred - solar_test_pred

bid_price = np.zeros(price_test_pred.shape)
bid_quantity = np.zeros(price_test_pred.shape)

charge_decision = charge_discharge(price_test_pred)
#charge_decision = np.zeros(price_test_pred.shape)

slice_values = np.array([15, 20, 25, 40, 60, 75, 90, 110, 160])

for i in range(bid_price.size):
	k_value = np.searchsorted(slice_values, actual_test_pred[i])
	bid_price[i] = (price_test_pred[i] + std_price[k_value] * x_values_price[k_value]).clip(max=7.)
	charge = (charge_decision[i] > 0).astype(np.int) * (5 + x_values_charging[k_value]*std_demand_quantity[k_value])
	discharge = (charge_decision[i] < 0).astype(np.int) * ((-4) + x_values_discharging[k_value]*std_demand_quantity[k_value])
	neutral = (charge_decision[i] == 0).astype(np.int) * (x_values_neutral[k_value]*std_demand_quantity[k_value])

	bid_quantity[i] = (actual_test_pred[i] + charge + discharge + neutral).clip(min=0)


final = np.vstack((bid_price, bid_quantity)).T
full_final = pd.DataFrame(final)
full_final.to_csv('7xx.csv', index=False)