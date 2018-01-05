import numpy as np
from another import black_box
import pandas as pd


x_value = 0.00902695
y_value = 0.14056995

with open('extracted_sigma.txt') as f:
	content = f.readlines()


demand_test_pred = pd.read_csv('Demand_LB_pred.csv', header=None).as_matrix()
solar_test_pred = pd.read_csv('Solar_LB_pred.csv', header=None).as_matrix()
price_test_pred = pd.read_csv('Price_LB_pred.csv', header=None).as_matrix()

quantity_test_pred = demand_test_pred = solar_test_pred

bid_price = np.zeros(price_test_pred.shape)
bid_quantity = np.zeros(price_test_pred.shape)

for hour in range(24):
	sigma_price = float(content[hour + 1].split()[0])
	sigma_quantity = float(content[hour + 1].split()[1])

	bid_price[:, hour] = price_test_pred[:, hour] + x_value * sigma_price
	bid_quantity[:, hour] = quantity_test_pred[:, hour] + y_value * sigma_quantity

quantity_star = black_box(bid_price.ravel(), bid_quantity.ravel())[1]

temp1 = np.reshape(bid_price, (np.product(bid_price.shape), 1))
temp2 = np.reshape(quantity_star, (np.product(bid_quantity.shape), 1))
final = np.concatenate((temp1, temp2), axis=1)
full_final = pd.DataFrame(final)
full_final.to_csv('7zz.csv', index=False)


