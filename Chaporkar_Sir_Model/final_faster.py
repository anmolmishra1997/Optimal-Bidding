import numpy as np
from another_faster import black_box
from cost_calculation import cost
import pandas as pd
import multiprocessing
from itertools import product
from multiprocessing.dummy import Pool as ThreadPool

pool = multiprocessing.Pool(4)

x_values = np.arange(-10, +10, 0.5)
y_values = np.arange(-10, +10, 0.5)

with open('extracted_sigma.txt') as f:
	content = f.readlines()

error_chart = np.zeros((x_values.size, y_values.size))

for block in range(17):
	print("Block ", block, " reached")
	demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
	demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]
	solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
	solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]
	price_train = pd.read_csv('Price_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
	price_train_pred = pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]

	quantity_train_pred = demand_train_pred - solar_train_pred
	quantity_train = demand_train - solar_train

	min_cost, temp = pool.map(black_box, np.concatenate((price_train.ravel(), quantity_train.ravel())))

	price_chart = np.zeros((x_values.size, y_values.size))

	for i in range(x_values.size):
		print("i value ", i)
		for j in range(y_values.size):
			print("j value ", j)
			for hour in range(24):
				sigma_price = float(content[25* block + hour + 1].split()[0])
				sigma_quantity = float(content[25*block + hour + 1].split()[1])

				quantity_train_pred[:, hour] = quantity_train_pred[:, hour] + x_values[i] * sigma_quantity
				price_train_pred[:, hour] = price_train_pred[:, hour] + y_values[j] * sigma_price
			#temp, quantity_star = black_box(price_train_pred.ravel(), quantity_train_pred.ravel())
			temp, quantity_star = pool.map(black_box, np.concatenate((price_train_pred.ravel(), quantity_train_pred.ravel())))
			price_chart[i, j] =  cost(demand_train, solar_train, price_train, quantity_train_pred, price_train_pred)
	#ERROR FUNCTION GOES HERE
	price_diff_chart = price_chart - min_cost
	error_chart += price_diff_chart

print(np.unravel_index(np.argmin(error_chart), (x_values.size, y_values.size)))

# LOAD TEST DATA
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

	temp, quantity_final[:, hour] = pool.map(black_box, np.concatenate((price_train_pred, quantity_train_pred)))

temp, quantity_star = black_box(price_train_pred.ravel(), quantity_train_pred.ravel())

print(cost(demand_train, solar_train, price_train, quantity_star, price_train_pred))
temp1 = np.reshape(price_train_pred, (np.product(price_train_pred.shape), 1))
temp2 = np.reshape(quantity_star, (np.product(quantity_star.shape), 1))
final = np.concatenate((temp1, temp2), axis=1)
full_final = pd.DataFrame(final)
full_final.to_csv('7zz.csv', index=False)