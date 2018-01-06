import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

demand_train = pd.read_csv('Demand_Train.csv', header=None).as_matrix()
demand_train_pred = pd.read_csv('Demand_Train_pred.csv', header=None).as_matrix()
solar_train = pd.read_csv('Solar_Train.csv', header=None).as_matrix()
solar_train_pred = pd.read_csv('Solar_Train_pred.csv', header=None).as_matrix()
price_train = pd.read_csv('Price_Train.csv', header=None).as_matrix()
price_train_pred = pd.read_csv('Price_Train_pred.csv', header=None).as_matrix()

x = np.arange(24)
plt.plot(x, [np.mean(demand_train[:, i]) for i in range(24)], axes={0,43,0,1})
plt.plot(x, [np.std(demand_train[:, i] - demand_train_pred[:, i]) for i in range(24)])
plt.show()

