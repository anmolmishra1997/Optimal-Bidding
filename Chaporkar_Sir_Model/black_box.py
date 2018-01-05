import numpy as np
import operator

def cost_one_battery_state_to_another(battery_state_left, battery_state_right, price, quantity):
	if battery_state_right > battery_state_left + 5:
		return np.inf
	if battery_state_right < battery_state_left - 5:
		return np.inf
	diff_states = battery_state_left - battery_state_right
	# If diff_states greater than 0 DISCHARGING
	# If diff_states less than 0 CHARGING
	if diff_states <= 0:
		return (np.abs(diff_states) + quantity) * price
	if diff_states > 0:
		return (-0.8*np.abs(diff_states) + quantity) * price

def find_lowest_cost(price, quantity, level):
	bid_quantity = np.zeros((26, quantity.size))
	# IF NOT LAST LEVEL
	if level != 0:
		last_price_column, bid_quantity[:, 1:] = find_lowest_cost(price[1:], quantity[1:], level-1)
		print(last_price_column)
		ans = np.full(26, np.inf)
		for i in range(26):
			temp = []
			for j in range(26):
				temp.append(last_price_column[j] + cost_one_battery_state_to_another(i, j, price[0], quantity[0]))
			index, ans[i] = min(enumerate(temp), key=operator.itemgetter(1))
			bid_quantity[i, 0] = index  - i
		return ans, bid_quantity
	# IF LAST LEVEL
	if level == 0:
		ans = np.full(26, np.inf)
		ans[0] = 0
		return ans, np.array([])