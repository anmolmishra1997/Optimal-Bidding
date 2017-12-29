import numpy as np
from scipy_optimize import charge_discharge
import pandas as pd

with open('pred_150.txt') as f:
	content = f.readlines()

std_price = []
x_values_price = []
std_demand_quantity = []
x_values_charging = []
x_values_discharging = []
x_values_neutral = []

for i in range(len(content)):
	if i % 5 == 0:
		std_price.append(float(content[i].split()[0]))
		std_demand_quantity.append(float(content[i].split()[1]))
	if i % 5 == 2:
		#Charging
		x_values_charging.append(float(content[i].split()[-2]))
		x_values_price.append(float(content[i].split()[-1]))
	if i % 5 == 3:
		x_values_discharging.append(float(content[i].split()[-2]))
	if i % 5 == 4:
		x_values_neutral.append(float(content[i].split()[-2]))


std_price = np.array(std_price)
x_values_price = np.array(x_values_price)
std_demand_quantity = np.array(std_demand_quantity)
x_values_charging = np.array(x_values_charging)
x_values_discharging = np.array(x_values_discharging)


demand_test_pred = pd.read_csv('Demand_LB_pred.csv', header=None).as_matrix()
solar_test_pred = pd.read_csv('Solar_LB_pred.csv', header=None).as_matrix()
price_test_pred = pd.read_csv('Price_LB_pred.csv', header=None).as_matrix()

bid_price = np.zeros(price_test_pred.shape)
bid_quantity = np.zeros(price_test_pred.shape)
charge_decision = charge_discharge(price_test_pred)

for hour in range(0, 24):
	bid_price[:, hour] = (price_test_pred[:, hour] + x_values_price[hour]*std_price[hour]).clip(max=7.)

	charge = (charge_decision[:, hour] > 0).astype(np.int) * (5 + x_values_charging[hour]*std_demand_quantity[hour])
	discharge = (charge_decision[:, hour] < 0).astype(np.int) * ((-4) + x_values_discharging[hour]*std_demand_quantity[hour])
	neutral = (charge_decision[:, hour] == 0).astype(np.int) * (x_values_neutral[hour]*std_demand_quantity[hour])
	bid_quantity[:, hour] = (demand_test_pred[:, hour] - solar_test_pred[:, hour] + charge + discharge + neutral).clip(min=0)


temp1 = np.reshape(bid_price, (np.product(bid_price.shape), 1))
temp2 = np.reshape(bid_quantity, (np.product(bid_quantity.shape), 1))
final = np.concatenate((temp1, temp2), axis=1)
full_final = pd.DataFrame(final)
full_final.to_csv('7zz.csv', index=False)