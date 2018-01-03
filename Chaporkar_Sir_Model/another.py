# You are given Price and Quantity array
import numpy as np
import operator
from black_box import cost_one_battery_state_to_another

def black_box(Price, Quantity):
	cost_matrix = np.full((26, Price.size + 1), np.inf)
	quantity_matrix = np.full((26, Price.size + 1), np.inf)
	cost_matrix[0, -1] = 0
	quantity_matrix[0, -1] = 0

	for hour in range(Price.size-1, -1, -1):
		for left in range(26):
			temp = []
			for right in range(26):
				temp.append(cost_matrix[right, hour + 1] + cost_one_battery_state_to_another(left, right, Price[hour], Quantity[hour]))
				#print cost_one_battery_state_to_another(left, right, Price[hour], Quantity[hour])
			quantity_matrix[left, hour], cost_matrix[left, hour] = min(enumerate(temp), key=operator.itemgetter(1))
	demand_list = [0.] + quantity_matrix[0, :].tolist()
	demand_list.pop()
	return cost_matrix[0,0], Quantity + demand_list[1:] - demand_list[:-1]
