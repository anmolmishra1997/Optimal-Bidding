import numpy as np
from scipy_optimize import charge_discharge
from cost_calculation import cost
import pandas as pd

with open('local_200.txt') as f:
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
x_values_neutral = np.array(x_values_neutral)


demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()[700:, :]
demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()[700:, :]
solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()[700:, :]
solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()[700:, :]
price_train = pd.read_csv('Price_Train.csv', header=None).as_matrix()[700:, :]
price_train_pred = pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()[700:, :]

bid_price = np.zeros(price_train_pred.shape)
bid_quantity = np.zeros(price_train_pred.shape)
charge_decision = charge_discharge(price_train_pred)


temp = np.zeros(demand_train.shape)

for hour in range(0, 24):
	bid_price[:, hour] = (price_train_pred[:, hour] + x_values_price[hour]*std_price[hour]).clip(max=7.)

	charge = (charge_decision[:, hour] > 0).astype(np.int) * (5 + x_values_charging[hour]*std_demand_quantity[hour])
	discharge = (charge_decision[:, hour] < 0).astype(np.int) * ((-4) + x_values_discharging[hour]*std_demand_quantity[hour])
	neutral = (charge_decision[:, hour] == 0).astype(np.int) * (x_values_neutral[hour]*std_demand_quantity[hour])
	bid_quantity[:, hour] = demand_train_pred[:, hour] - solar_train_pred[:, hour] + charge + discharge
	#temp[:, hour] = demand_train[:, hour] - solar_train[:, hour] + charge + discharge + neutral
	temp[:, hour] = demand_train[:, hour] - solar_train[:, hour] + 5*(charge_decision[:, hour] > 0).astype(np.int) - 4*(charge_decision[:, hour] < 0).astype(np.int)


print cost(demand_train, solar_train, price_train, bid_quantity, bid_price)

print cost(demand_train, solar_train, price_train, demand_train - solar_train, price_train)

print cost(demand_train, solar_train, price_train, temp, price_train)
