import numpy as np
from math import sqrt
from cvxopt import matrix
from cvxopt.solvers import qp

def charge_discharge(Price):
	price = Price.ravel()

	P = 2*matrix(np.diag(price), tc='d')
	q = 9*matrix(np.array(price), tc='d')

	G = matrix(np.concatenate((np.tril(np.ones(price.size)), -1*np.tril(np.ones(price.size)), -1*np.identity(price.size), np.identity(price.size))), tc='d')
	H = matrix(np.concatenate(((np.arange(price.size)+1).clip(max=5), np.zeros(price.size), np.ones(2*price.size))), tc='d')

	sol = qp(P, q, G, H)
	print np.sum(np.around(np.array(sol['x']).reshape(Price.shape)).clip(min=-0.8) * Price) < 0
	return  np.around(np.array(sol['x']).reshape(Price.shape))


#print charge_discharge(np.array([1,5,4,3,7,6,8]))