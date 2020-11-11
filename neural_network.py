import numpy as np
import matplotlib.pyplot as plt
import random

class NeuralNetwork(object):
	"""docstring for NeuralNetwork"""
	def __init__(self, Inputs=1,hidden=1,output=1):
		super(NeuralNetwork, self).__init__()
		#layers
		self.input=Inputs
		self.hidden=hidden
		self.output=output
		
		#weights between layer1 and layer2
		self.w1=self.init_weight(self.input,self.hidden)

		#weights between layer2 and layer2
		self.w2=self.init_weight(self.hidden,self.output)

		#biases between layer1 and layer2
		self.b1=np.matrix([random.randint(-8,8)/10 for j in range(self.hidden)]).T

		#biases between layer2 and layer3
		self.b2=np.matrix([random.randint(-8,8)/10 for j in range(self.output)]).T

		#learning rate
		self.learning_rate=0.1


	def init_weight(self,input,output):
		"""initializing random weights"""
		arr=[[random.randint(-8,8)/10 for j in range(input)] for i in range(output)]
		return np.matrix(arr)
	
	def print_weights(self):
		print(self.w1)
		print(self.w2)

	def sigmoid(self,x):
		return 1/(1 + np.exp(-x) )

	def sigmoid_p(self,x):
		p = self.sigmoid(x)
		return np.multiply(p ,(1-p))


	def feedForward(self,input):
		l1= np.matrix(input).T
		l2= self.w1*l1
		l2+=self.b1+l2
		l2 = self.sigmoid(l2)
		l3= self.w2 *l2
		l3+=self.b2+l3
		l3 = self.sigmoid(l3)
		return l3

	def predict(self,input):
		ans_mtx=self.feedForward(input)
		output=np.array(ans_mtx.T)
		return output[0] 

	def error(self,Input,y):
		y_hat =self.predict(Input)
		return (y-y_hat)**2 /2

	def backProp(self,inputs,y):
		y   = np.matrix(y).T
		l1  = np.matrix(inputs).T
		l2  = self.w1*l1
		l2 += self.b1+l2
		l2  = self.sigmoid(l2)
		l3  = self.w2 *l2
		l3 += self.b2+l3
		l3  = self.sigmoid(l3)

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
		self.w1 += delta1

		#         changing second biases
		#         (---cross product-----)    * learning_rate 
		delta1_b =  (np.multiply(dEdA,dAdZ)) * self.learning_rate
		self.b1 += delta1_b

	def train(self,x,y,iteration=1000):
		n = len(x)
		x=np.array(x)
		y=np.array(y)
		for _ in range(iteration):
			index = random.randint(0,n-1)
			self.backProp(x[index],y[index])




if __name__ == '__main__':

	brain = NeuralNetwork(2,5,1)

	# brain.print_weights()	

	print("##")
	print(brain.predict([1,1]))
	# print(sum(brain.error([0,0],[0])))
	print("##")

	data =     [[[0,0],[0]],
				[[1,1],[0]],
				[[1,0],[1]],
				[[0,1],[1]]]

	# brain.backProp([1,1],[0])
	x = [item[0] for item in data]
	y = [item[1] for item in data]
	print(x)
	print(y)
	brain.train(x,y)
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