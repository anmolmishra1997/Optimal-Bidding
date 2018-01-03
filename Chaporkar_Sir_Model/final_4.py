import numpy as np
from another import black_box
from cost_calculation import cost
import pandas as pd

block = 4

x_values = np.arange(-3, +4, 0.2)
y_values = np.arange(-2, +4, 0.2)

with open('extracted_sigma.txt') as f:
	content = f.readlines()

demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]
solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]
price_train = pd.read_csv('Price_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
price_train_pred = pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]

quantity_train_pred = demand_train_pred - solar_train_pred
quantity_train = demand_train - solar_train

min_cost, temp = black_box(price_train.ravel(), quantity_train.ravel())

price_chart = np.zeros((x_values.size, y_values.size))

for i in range(x_values.size):
	print "i value ", i
	for j in range(y_values.size):
		for hour in range(24):
			sigma_price = float(content[25* block + hour + 1].split()[0])
			sigma_quantity = float(content[25*block + hour + 1].split()[1])

			quantity_train_pred[:, hour] = quantity_train_pred[:, hour] + x_values[i] * sigma_quantity
			price_train_pred[:, hour] = price_train_pred[:, hour] + y_values[j] * sigma_price
		temp, quantity_star = black_box(price_train_pred.ravel(), quantity_train_pred.ravel())
		price_chart[i, j] =  cost(demand_train, solar_train, price_train, quantity_star.reshape(quantity_train_pred.shape), price_train_pred)
#ERROR FUNCTION GOES HERE
price_diff_chart = price_chart - min_cost

np.savetxt('block_{}.txt'.format(block), price_diff_chart, fmt='%.3e')

