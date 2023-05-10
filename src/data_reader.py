import numpy as np 
import matplotlib.pyplot as plt

# Class,User,X0,Y0,Z0,X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4,X5,Y5,Z5,X6,Y6,Z6,X7,Y7,Z7,X8,Y8,Z8,X9,Y9,Z9,X10,Y10,Z10,X11,Y11,Z11
# 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

RODADAS = 10

def get_data():
	Data = np.genfromtxt("data/data.csv", dtype="str", delimiter =",") 
	seed = np . random . permutation ( Data . shape [0]) 
	X = Data [ seed ,2:] 
	Y = Data [ seed ,0:1] 
	 
	Xtreino = X [0: int( X. shape [0]*.8) ,:] 
	Ytreino = Y [0: int( X. shape [0]*.8) ,:] 
	
	Xteste = X[ int (X. shape [0]*.8) : ,:] 
	Yteste = Y[ int (X. shape [0]*.8) : ,:] 	

	return Xtreino, Ytreino, Xteste, Yteste