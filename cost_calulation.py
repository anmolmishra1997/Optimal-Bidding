def cost(demand, solar, price, bid_quantity, bid_price):
	'''
	All the parameters here are in NumPy array.
	This function evaluates cost
	To use this function in other Python code, use from cost_calculation import cost
	'''
	cost = 0
	battery = 0
	for i in range(0, demand.shape[0]):
		for j in range(0, demand.shape[1]):
			bid_won = (bid_price[i][j] >= price[i][j])
			if not bid_won:
				bid_quantity[i][j] = 0
			cost += bid_quantity[i][j] * price[i][j] + max(0, 7*(demand[i][j] - bid_quantity[i][j] - solar[i][j] - min(0.8*battery, 4)))
			if solar[i][j] + bid_quantity[i][j] > demand[i][j]:
				battery = min(25, battery + min(5, solar[i][j] + bid_quantity[i][j] - demand[i][j]))
			else:
				battery = battery - min(battery, 4)
	return cost
