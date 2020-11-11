import numpy as np
import matplotlib.pyplot as plt
import random

class NeuralNetwork(object):
	"""docstring for NeuralNetwork"""
	def __init__(self, layers):
		super(NeuralNetwork, self).__init__()
		#layers
		self.layers=layers

		#weights
		self.weights = self.get_weights(self.layers)

		#biases
		self.biases = self.get_biases(self.layers)

		#learning rate
		self.learning_rate=0.02


	def init_weight(self,input,output):
		"""initializing random weights"""
		arr=[[random.randint(-8,8)/10 for j in range(input)] for i in range(output)]
		return np.matrix(arr)

	def get_weights(self,layers):
		weights=[]
		for i in range(1,len(layers)):
			w = self.init_weight(layers[i-1],layers[i])
			weights.append(w)
		return weights

	def get_biases(self,layers):
		biases=[]
		for i in range(1,len(layers)):
			b=np.matrix([random.randint(-8,8)/10 for j in range(layers[i])]).T
			biases.append(b)
		return biases
	
	def print_weights(self):
		for i in range(len(self.weights)):
			print("-----------")
			print(f"weights {i+1}")
			print("-----------")
			print(self.weights[i])
		print("-----------")

	def sigmoid(self,x):
		return 1/(1 + np.exp(-x) )

	def sigmoid_p(self,x):
		p = self.sigmoid(x)
		return np.multiply(p ,(1-p))


	def feedForward(self,inputs):
		tensor = np.matrix(inputs).T
		for i in range(len(self.weights)):
			tensor  = self.weights[i]*tensor
			tensor += self.biases[i]
			tensor  = self.sigmoid(tensor)
		return tensor

	def predict(self,input):
		ans_mtx=self.feedForward(input)
		output=np.array(ans_mtx.T)
		return output[0] 

	def error(self,Input,y):
		y_hat =self.predict(Input)
		return (y-y_hat)**2 /2

	def backProp(self,inputs,y):
		y=np.matrix(y).T

		tensor = np.matrix(inputs).T
		layers_tensors=[tensor]

		for i in range(len(self.weights)):
			tensor  = self.weights[i]*tensor
			tensor += self.biases[i]
			tensor  = self.sigmoid(tensor)
			layers_tensors.append(tensor)

		layers_tensors_r = layers_tensors[::-1]

		errors=[y-tensor]
		n=len(self.weights)
		for i in range(1,n):
			error=self.weights[n-i].T * errors[-1]
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
			self.weights[n-1-i] += delta

			#         changing second biases
			#         (---cross product-----)    * learning_rate 
			delta_b =  (np.multiply(dEdA,dAdZ)) * self.learning_rate
			self.biases[n-1-i] += delta_b
		return
		# return tensor
		# pred =      np.matrix(self.predict(inputs)).T
		#de/dw = de/da * da/dz * dz/dw
		#gredients
		dEdA = 		y-l3
		dAdZ =      np.multiply(l3,(1-l3)) # sigmoid_prime (derivative)
		dZdW =      l2

		#         changing second weights
		#         (------------dot product------) * learning_rate  
		#         (---cross product-----) |		  |
		delta2 =  (np.multiply(dEdA,dAdZ) * dZdW.T) * self.learning_rate
		# print(delta2)
		self.w2 += delta2

		#         changing second biases
		#         (---cross product-----)    * learning_rate 
		delta2_b =  (np.multiply(dEdA,dAdZ)) * self.learning_rate
		self.b2 += delta2_b


		#   			hidden error
		#				W2 tranpose * error (from first layer)   
		hidden_error =  self.w2.T   * (y-l3)

		#de/dw = de/da * da/dz * dz/dw
		#gredients
		dEdA = 		hidden_error
		dAdZ =      np.multiply(l2,(1-l2)) # sigmoid_prime (derivative)
		dZdW =      l1                       #input
		#         changing first weights
		#         (------------dot product------) * learning_rate  
		#         (---cross product-----) |		  |
		delta1   =  (np.multiply(dEdA,dAdZ) * dZdW.T) * self.learning_rate
		# print(delta1)
		self.w1 -= delta1

		#         changing second biases
		#         (---cross product-----)    * learning_rate 
		delta1_b =  (np.multiply(dEdA,dAdZ)) * self.learning_rate
		self.b1 -= delta1_b


	def train(self,x,y,iteration=1000,lr_r=0.95):
		n = len(x)
		x=np.array(x)
		y=np.array(y)
		for i in range(iteration):
			index = random.randint(0,n-1)
			self.backProp(x[index],y[index])
			# if i%1000==0:
			# 	self.learning_rate*=lr_r




if __name__ == '__main__':

	brain = NeuralNetwork([2,5,1])

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

	print("##")
	print(brain.predict([1,1]))
	# print(sum(brain.error([0,0],[0])))


	while True:
		inp = input(":")
		if(inp=="x"):
			break
		lst = [int(i) for i in inp.split(" ")]

		print(brain.predict(lst[:2]))

	# # brain.draw_sigmoid()