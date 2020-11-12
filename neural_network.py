import numpy as np
import matplotlib.pyplot as plt
import random

class NeuralNetwork(object):
	"""docstring for NeuralNetwork"""
	def __init__(self, layers,learning_rate=0.2):
		super(NeuralNetwork, self).__init__()
		#layers
		self.layers=layers

		#weights
		self.weights = self.get_weights(self.layers)

		#biases
		self.biases = self.get_biases(self.layers)

		#learning rate
		self.learning_rate=learning_rate


	def init_weight(self,input,output):
		"""initializing random weights"""
		arr=[[random.randint(-8,8)/10 for j in range(input)] for i in range(output)]
		return np.matrix(arr)

	def get_weights(self,layers):
		'''return list of randomly generated weights'''
		weights=[]
		for i in range(1,len(layers)):
			w = self.init_weight(layers[i-1],layers[i])
			weights.append(w)
		return weights

	def get_biases(self,layers):
		'''return list of randomly generated biases'''
		biases=[]
		for i in range(1,len(layers)):
			b=np.matrix([random.randint(-8,8)/10 for j in range(layers[i])]).T
			biases.append(b)
		return biases
	
	def print_parameters(self):
		for i in range(len(self.weights)):
			print("-----------")
			print(f"weights layer: {i+1,i+2}")
			print("-----------")
			print(self.weights[i])
			print("-----------")
			print(f"biases layers: {i+2}")
			print("-----------")
			print(self.biases[i])
		print("-----------")

	def get_parameters(self):
		'''return weights and biases '''
		'''   return type tuple      '''
		'''   (weights,bieses)       '''
		return (self.weights,self.biases)

	def sigmoid(self,x):
		'''non linear activation function'''
		return 1/(1 + np.exp(-x) )

	def sigmoid_p(self,x):
		'''derivative of sigmoid'''
		p = self.sigmoid(x)
		return np.multiply(p ,(1-p))


	def feedForward(self,inputs):
		'''return prediction vector'''
		tensor = np.matrix(inputs).T
		for i in range(len(self.weights)):
			tensor  = self.weights[i]*tensor
			tensor += self.biases[i]
			tensor  = self.sigmoid(tensor)
		return tensor

	def predict(self,input):
		'''prediction for given input'''
		'''return numpy.array of predition'''
		ans_mtx=self.feedForward(input)
		'''convering matrix in 1d numpy array'''
		output=np.array(ans_mtx.T)
		return output[0] 

	def error(self,Input,y):
		'''mean squared error'''
		y_hat =self.predict(Input)
		return (y-y_hat)**2 /2

	def backProp(self,inputs,y):
		y=np.matrix(y).T

		tensor = np.matrix(inputs).T
		layers_tensors=[tensor]
		#feedForword
		#saving activation in layers_tensors
		for i in range(len(self.weights)):
			tensor  = self.weights[i]*tensor
			tensor += self.biases[i]
			tensor  = self.sigmoid(tensor)
			layers_tensors.append(tensor)

		#reverse tensor layers
		layers_tensors_r = layers_tensors[::-1]

		#error at each layer
		errors=[y-tensor]
		n=len(self.weights)
		for i in range(1,n):
			#     weights between layer (n-1 , n)    error at layer (n)
			error=   self.weights[n-i].T           *  errors[-1]
			errors.append(error)

		for i in range(n):
			#de/dw = de/da * da/dz * dz/dw
			#gredients
			dEdA = 		errors[i]
			dAdZ =      np.multiply(layers_tensors_r[i],(1-layers_tensors_r[i])) # sigmoid_prime (derivative)
			dZdW =      layers_tensors_r[i+1]

			#         changing second weights
			#         (------------dot product------) * learning_rate  
			#         (---cross product-----) |		  |
			delta =  (np.multiply(dEdA,dAdZ) * dZdW.T) * self.learning_rate

			#updating weights with delta
			self.weights[n-1-i] += delta


			#         changing second biases
			#         (---cross product-----)    * learning_rate 
			delta_b =  (np.multiply(dEdA,dAdZ)) * self.learning_rate

			#updating weights with delta
			self.biases[n-1-i] += delta_b
		return


	def train(self,x,y,iteration=10000,lr_r=0.95):
		n = len(x)
		x=np.array(x)
		y=np.array(y)
		for i in range(iteration):
			index = random.randint(0,n-1)
			self.backProp(x[index],y[index])
			# if i%1000==0:
			# 	self.learning_rate*=lr_r




if __name__ == '__main__':

	brain = NeuralNetwork([2,5,1],0.1)

	# brain.print_weights()	

	print("##")
	print(brain.predict([1,1]))
	# print(sum(brain.error([0,0],[0])))
	print("##")

	data =     [[[0,0],[0]],
				[[1,1],[0]],
				[[1,0],[1]],
				[[0,1],[1]]]

	brain.backProp([1,1],[0])
	x = [item[0] for item in data]
	y = [item[1] for item in data]
	# print(x)
	# print(y)
	brain.train(x,y,10000)
	# for i in range(10000):
	# 	item  = random.choice(data)
	# 	brain.backProp(item[0],item[1])
	# 	# brain.backProp([1,1],[0])
	# brain.print_parameters()
	print("##")
	print(brain.predict([1,1]))
	# print(sum(brain.error([0,0],[0])))


	while True:
		inp = input(":")
		if(inp=="x"):
			break
		if(inp=="t"):
			brain.train(x,y,10000)
			# brain.print_parameters()
			print(brain.predict([1,1]))
			continue

		lst = [int(i) for i in inp.split(" ")]

		print(brain.predict(lst[:2]))

	# # brain.draw_sigmoid()