import numpy as np
from binary_classification import charge_discharge
from cost_calulation import cost
import pandas as pd

with open('bid_price_prediction.txt') as f:
	content = f.readlines()

std_price = []
x_values_price = []

for i in range(len(content)):
	if i % 3 == 0:
		std_price.append(float(content[i]))
	if i % 3 == 2:
		x_values_price.append(float(content[i].split()[-1]))

std_demand_quantity = []
x_values_charging = []
x_values_discharging = []

with open('bid_quantity_prediction.txt') as f:
	content = f.readlines()

for i in range(len(content)):
	if i % 4 == 0:
		std_demand_quantity.append(float(content[i]))
	if i % 4 == 2:
		#Charging
		x_values_charging.append(float(content[i].split()[-1]))
	if i % 4 == 3:
		x_values_discharging.append(float(content[i].split()[-1]))

std_price = np.array(std_price)
x_values_price = np.array(x_values_price)
std_demand_quantity = np.array(std_demand_quantity)
x_values_charging = np.array(x_values_charging)
x_values_discharging = np.array(x_values_discharging)

demand_test = pd.read_csv('Demand_Train.csv', header=None).as_matrix()[700:, :]
demand_test_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()[700:, :]
solar_test = pd.read_csv('Solar_Train.csv', header=None).as_matrix()[700:, :]
solar_test_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()[700:, :]
price_test = pd.read_csv('Price_Train.csv', header=None).as_matrix()[700:, :]
price_test_pred = pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()[700:, :]

bid_price = np.zeros(price_test_pred.shape)
bid_quantity = np.zeros(price_test_pred.shape)
charge_decision = charge_discharge(price_test_pred)

for hour in range(0, 24):
	bid_price[:, hour] = price_test_pred[:, hour] + x_values_price[hour]*std_price[hour]
	print x_values_price[hour]*std_price[hour]

	charge = (charge_decision[:, hour] > 0).astype(np.int) * (5 + x_values_charging[hour]*std_demand_quantity[hour])
	discharge = (charge_decision[:, hour] < 0).astype(np.int) * ((-4) + x_values_discharging[hour]*std_demand_quantity[hour])

	bid_quantity[:, hour] = demand_test_pred[:, hour] + charge + discharge

print cost(demand_test, solar_test, price_test, bid_quantity, bid_price)

print cost(demand_test, solar_test, price_test, demand_test_pred, price_test_pred)

print cost(demand_test, solar_test, price_test, demand_test, price_test)

print cost(demand_test, solar_test, price_test, bid_quantity, price_test)





