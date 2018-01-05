import numpy as np
from another import black_box
from cost_calculation import cost
import pandas as pd
from pyswarm import pso

with open('extracted_sigma.txt') as f:
	content = f.readlines()

demand_train = []
demand_train_pred = []
solar_train = []
solar_train_pred = []
price_train = []
price_train_pred = []

for block in range(17):
	demand_train.append(pd.read_csv('Demand_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :])
	demand_train_pred.append(pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :])
	solar_train.append(pd.read_csv('Solar_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :])
	solar_train_pred.append(pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :])
	price_train.append(pd.read_csv('Price_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :])
	price_train_pred.append(pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :])

def minimize_func(x):
	x_value = x[0]
	y_value = x[1]
	minimize_variable = 0
	for block in range(10):
		quantity_train_pred = demand_train_pred[block] - solar_train_pred[block]
		quantity_train = demand_train[block] - solar_train[block]

		min_cost = black_box(price_train[block].ravel(), quantity_train.ravel())[0]

		for hour in range(24):
			sigma_price = float(content[25* block + hour + 1].split()[0])
			sigma_quantity = float(content[25*block + hour + 1].split()[1])

			quantity_train_pred[:, hour] = quantity_train_pred[:, hour] + x_value * sigma_quantity
			price_train_pred[block][:, hour] = price_train_pred[block][:, hour] + y_value * sigma_price
		quantity_star = black_box(price_train_pred[block].ravel(), quantity_train_pred.ravel())[1]
		price_chart = cost(demand_train[block], solar_train[block], price_train[block], quantity_star.reshape(price_train_pred[block].shape) , price_train_pred[block])
		error = price_chart - min_cost  #THIS IS THE ERROR FUNCTION, ABSOLUTE HERE

		minimize_variable += error
	return minimize_variable

lb = [-5, -5]
ub = [5, 5]
xopt, fopt = pso(minimize_func, lb, ub, swarmsize = 20, maxiter = 5, debug = True)

print( xopt)
print(fopt)



############################## LOAD TEST DATA ##################################
'''
block = 17
demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]
solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]
price_train = pd.read_csv('Price_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
price_train_pred = pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]

quantity_train_pred = demand_train_pred - solar_train_pred
quantity_final = np.zeros(quantity_train_pred.shape)

for hour in range(24):
	(mu, sigma_quantity) = norm.fit(quantity_train_pred[:, hour])
	(mu, sigma_price) = norm.fit(price_train_pred[:, hour])
	price_train_pred[:, hour] = price_train_pred[:, hour] + np.unravel_index(np.argmin(error_chart), (x_values.size, y_values.size))[0] * sigma_price
	quantity_train_pred[:, hour] = quantity_train_pred[:, hour] + np.unravel_index(np.argmin(error_chart), (x_values.size, y_values.size))[1] * sigma_quantity

	temp, quantity_final[:, hour] = black_box(price_train_pred, quantity_train_pred)

temp, quantity_star = black_box(price_train_pred.ravel(), quantity_train_pred.ravel())

print cost(demand_train, solar_train, price_train, quantity_star, price_train_pred)
print cost(demand_train, solar_train, price_train, black_box(price_train_pred, demand_train - solar_train)[1], price_train)
'''
