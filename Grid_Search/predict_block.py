import numpy as np
from scipy_optimize import charge_discharge
import pandas as pd
from another import black_box_quantity

with open('temp.txt') as f:
	content_blocks = f.readlines()

with open('test_900_block.txt') as f:
	content = f.readlines()

std_price = []
x_values_price = []
std_demand_quantity = []
x_values_charging = []
x_values_discharging = []
x_values_neutral = []

var = [std_demand_quantity, std_price, x_values_charging, x_values_discharging, x_values_neutral, x_values_price]
for i in range(6):
	for j in range(72):
		var[i].append(float(content[i].split()[j]))
print len(x_values_price)
print std_price
print x_values_price


demand_test_pred = pd.read_csv('Demand_LB_pred.csv', header=None).as_matrix()
solar_test_pred = pd.read_csv('Solar_LB_pred.csv', header=None).as_matrix()
price_test_pred = pd.read_csv('Price_LB_pred.csv', header=None).as_matrix()

bid_price = np.zeros(price_test_pred.shape)
bid_quantity = np.zeros(price_test_pred.shape)
#charge_decision = black_box_quantity(price_test_pred.ravel(), (demand_test_pred - solar_test_pred).ravel())
#print charge_decision.shape
#print type(charge_decision)
#charge_decision = charge_decision.reshape(price_test_pred.shape)
charge_decision = np.zeros(price_test_pred.shape)

for hour in range(0, 24):
	print "HOUR ", hour
	effective_demand = demand_test_pred[:, hour] - solar_test_pred[:, hour]
	deviation_price = np.zeros(effective_demand.shape)
	deviation_quantity = np.zeros(effective_demand.shape)
	for foo in range(deviation_price.size):
		if effective_demand[foo] <= float(content_blocks[3*hour]):
			deviation_price[foo] = std_price[3*hour] * x_values_price[3*hour]
			deviation_quantity[foo] = std_demand_quantity[3*hour] * x_values_neutral[3*hour]
		if effective_demand[foo] > float(content_blocks[3*hour]) and effective_demand[foo] <= float(content_blocks[3*hour + 1]):
			deviation_price[foo] = std_price[3*hour + 1] * x_values_price[3*hour + 1]
			deviation_quantity[foo] = std_demand_quantity[3*hour + 1] * x_values_neutral[3*hour + 1]
		if effective_demand[foo] > float(content_blocks[3*hour + 1]):
			deviation_price[foo] = std_price[3*hour + 2] * x_values_price[3*hour + 2]
			deviation_quantity[foo] = std_demand_quantity[3*hour + 2] * x_values_neutral[3*hour + 2]

	bid_price[:, hour] = price_test_pred[:, hour] + deviation_price
	bid_quantity[:, hour] = demand_test_pred[:, hour] - solar_test_pred[:, hour] + deviation_quantity

	#bid_price[:, hour] = (price_test_pred[:, hour] + smallest_set*x_values_price[3*hour + 0] + middle_set*x_values_price[3*hour + 1] + largest_set*x_values_price[3*hour + 2]).clip(max=7.)
	#charge = (charge_decision[:, hour] > 0).astype(np.int) * (5 + x_values_charging[hour]*std_demand_quantity[hour])           #NOT CHANGED
	#discharge = (charge_decision[:, hour] < 0).astype(np.int) * ((-4) + x_values_discharging[hour]*std_demand_quantity[hour])   #NOT CHANGED
	#neutral = (charge_decision[:, hour] == 0).astype(np.int) * (x_values_neutral[hour]*std_demand_quantity[hour])   #LEFT
	#bid_quantity[:, hour] = (demand_test_pred[:, hour] - solar_test_pred[:, hour] + charge + discharge + neutral).clip(min=0)




temp1 = np.reshape(bid_price, (np.product(bid_price.shape), 1))
temp2 = np.reshape(bid_quantity, (np.product(bid_quantity.shape), 1))
temp2 = np.reshape(black_box_quantity(temp1.ravel(), temp2.ravel())[1], (np.product(bid_quantity.shape), 1))
final = np.concatenate((temp1, temp2), axis=1)
full_final = pd.DataFrame(final)
full_final.to_csv('7zzz.csv', index=False)