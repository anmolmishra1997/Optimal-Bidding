import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

sigma_qty = []
sigma_price = []
x_values_qty_charging = []
x_values_qty_discharging = []
x_values_qty_neutral = []
x_values_price = []

for block in range(18):
	print "Block ", block
	demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
	demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]
	solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
	solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]
	price_train = pd.read_csv('Price_Train.csv', header=None).as_matrix()[50*block:50*(block+1), :]
	price_train_pred = pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()[50*block:50*(block+1), :]

	x_values = np.arange(-10, +10, 0.1)
	y_values = np.arange(-10, +10, 0.1)
	cost_charging = np.zeros((x_values.size, y_values.size))
	cost_discharging = np.zeros((x_values.size, y_values.size))
	cost_neutral = np.zeros((x_values.size, y_values.size))

	temp_qty_sigma = []
	temp_price_sigma = []
	temp_x_value_qty_charging = []
	temp_x_value_qty_discharging = []
	temp_x_value_qty_neutral = []
	temp_x_value_price = []

	for hour in range(24):

		error_data_quantity = (demand_train[:, hour] - solar_train[:, hour]) - (demand_train_pred[:, hour] - solar_train_pred[:, hour])
		error_data_price = (price_train_pred[:, hour] - price_train[:, hour])

		(mu_error_quantity, sigma_error_quantity) = norm.fit(error_data_quantity)
		(mu_error_price, sigma_error_price) = norm.fit(error_data_price)

		print sigma_error_price, sigma_error_quantity