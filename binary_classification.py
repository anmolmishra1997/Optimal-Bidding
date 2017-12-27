import numpy as np

def decide_charge_discharge(Price):
	print Price.shape
	print Price.size
	answer = np.zeros(Price.shape, dtype=bool)
	max = []
	max_indices = []
	# Reshape 2d into 1d
	sorted = np.sort(Price)
	sorted_indices = np.argsort(Price)
	while(answer.sum() <= (Price.size + 1)/2.):
		for i in range(0, sorted.size):
			index = sorted_indices[i]
			neighbors = range(index-4,index+5)
			neighbors.remove(index)
			good = False
			for q in range(0, 5):
				temp = set(neighbors[q:4+q]) < set(max_indices)
				good = good + temp
			if not good:
				max.append(sorted[i])
				max_indices.append(sorted_indices[i])
				answer[index] = True
	return answer

#print decide_charge_discharge(np.array([1,5,4,3,7,6,3,5,6,9,1]))

def charge_discharge(Price):
	# Correct version
	boolean = Price < np.median(Price)
	decision = boolean.astype(int)
	count=0
	for i in range(0, decision.shape[0]):
		for j in range(0, decision.shape[1]):
			if decision[i, j] == 1:
				count += 1
			if decision[i, j] == 0:
				count -= 1
				decision[i, j] = -1
			if count > 5:
				count -= 1
				decision[i, j] = 0
			if count < 0:
				count += 1
				decision[i, j] = 0
	return decision

print charge_discharge(np.array([[1,5,4,3,7],[6,3,5,6,9]]))
