import numpy as np
import pandas as pd
from another import black_box

with open('extracted_sigma.txt') as f:
	content = f.readlines()


x_values = np.arange(-3, +4, 0.2)
y_values = np.arange(-2, +4, 0.2)

a = np.loadtxt('block_1.txt')
print x_values[np.unravel_index(np.argmin(a), a.shape)[0]]
print y_values[np.unravel_index(np.argmin(a), a.shape)[1]]
b = np.loadtxt('block_2.txt')
print x_values[np.unravel_index(np.argmin(b), b.shape)[0]]
print y_values[np.unravel_index(np.argmin(b), b.shape)[1]]
c = np.loadtxt('block_3.txt')
print x_values[np.unravel_index(np.argmin(c), b.shape)[0]]
print y_values[np.unravel_index(np.argmin(c), b.shape)[1]]
d = np.loadtxt('block_4.txt')
print x_values[np.unravel_index(np.argmin(d), d.shape)[0]]
print y_values[np.unravel_index(np.argmin(d), d.shape)[1]]
e = np.loadtxt('block_5.txt')
print x_values[np.unravel_index(np.argmin(e), e.shape)[0]]
print y_values[np.unravel_index(np.argmin(e), e.shape)[1]]

ans = a+b+c+d+e

x = x_values[np.unravel_index(np.argmin(ans), ans.shape)[0]]
y = y_values[np.unravel_index(np.argmin(ans), ans.shape)[1]]


demand_train_pred = pd.read_csv('Demand_LB_pred.csv', header=None).as_matrix()
solar_train_pred = pd.read_csv('Solar_LB_pred.csv', header=None).as_matrix()
price_train_pred = pd.read_csv('Price_LB_pred.csv', header=None).as_matrix()

quantity_train_pred = demand_train_pred - solar_train_pred

for hour in range(24):
	sigma_price = float(content[hour + 1].split()[0])
	sigma_quantity = float(content[hour + 1].split()[1])

	quantity_train_pred[:, hour] = quantity_train_pred[:, hour] + x*sigma_quantity
	price_train_pred[:, hour] = price_train_pred[:, hour] + y*sigma_price

quantity_star = black_box(price_train_pred.ravel(), quantity_train_pred.ravel())[1]

temp1 = np.reshape(price_train_pred, (np.product(price_train_pred.shape), 1))
temp2 = np.reshape(quantity_star, (np.product(quantity_star.shape), 1))
final = np.concatenate((temp1, temp2), axis=1)
full_final = pd.DataFrame(final)
full_final.to_csv('7zz.csv', index=False)

