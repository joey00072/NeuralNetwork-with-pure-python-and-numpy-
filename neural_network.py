import numpy as np
import matplotlib.pyplot as plt
import random

class NeuralNetwork(object):
	"""docstring for NeuralNetwork"""
	def __init__(self, Inputs=1,hidden=1,output=1):
		super(NeuralNetwork, self).__init__()
		self.input=Inputs
		self.hidden=hidden
		self.output=output
		self.w1=self.init_weight(self.input,self.hidden)
		self.w2=self.init_weight(self.hidden,self.output)
		self.l1=None
		self.l2=None
		self.l3=None
		self.b2=np.matrix([random.randint(-8,8)/10 for j in range(self.hidden)]).T
		self.b3=np.matrix([random.randint(-8,8)/10 for j in range(self.output)]).T
		self.learning_rate=0.2
		# print(self.w1)

	def init_weight(self,input,output):
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

	def modify_for_bias(self,mtx):
		mul=np.matrix([[1,0,0],
						[0,1,0],
						[0,0,1],
						[0,0,0]])
		mtx= mul*mtx
		add=np.matrix([ [0],
						[0],
						[0],
					    [1]])
		return mtx+add

	def predict(self,input):
		l1= np.matrix(input).T
		self.l1 = l1
		# print(l1)
		# print("---") 
		# print(self.w1.shape)
		l2= self.w1*l1
		# print(l2)
		l2+=self.b2+l2
		l2 = self.sigmoid(l2)
		self.l2=l2
		# print("l2",self.w2)
		l3= self.w2 *l2
		l3+=self.b3+l3
		l3 = self.sigmoid(l3)
		self.l3 = l3
		# print("l3",l3)
		l3=np.array(l3.T)
		return l3[0]

	def error(self,Input,y):
		y_hat =self.predict(Input)
		return (y-y_hat)**2 /2

	def backProp(self,inputs,y):
		#de/dw = de/da * da/dz * dz/dw
		y=np.matrix(y).T
		# print(y)
		pred= np.matrix(self.predict(inputs)).T
		# print(pred)
		dEdA = 		y-pred 
		dAdZ =      np.multiply(self.l3,(1-self.l3))
		dZdW =      self.l2
		# print(np.multiply(dEdA,dAdZ).shape,self.b3.shape)
		delta =np.multiply(dEdA,dAdZ)* dZdW.T *self.learning_rate
		self.w2+=delta

		delta_b =np.multiply(dEdA,dAdZ)*self.learning_rate
		self.b3+=delta_b
		# hidden_error = self.w2.T*self.l2
		# print("..")
		# print(dEdA)
		# print("..")
		# print(y)
		# print(self.w2)
		# print("..")
		# print(pred)
		# print("..")
		hidden_error= self.w2.T*dEdA
		# print(hidden_error)
		dEdA2= hidden_error
		dAdZ2 = np.multiply(self.l2,(1-self.l2))
		dZdW2 = inputs
		# print(np.multiply(dEdA2,dAdZ2),np.matrix(inputs))
		hidden_delta = np.multiply(dEdA2,dAdZ2)*y.T*self.learning_rate
		self.w1+=hidden_delta
		# print(self.w1)
		hidden_delta_b = np.multiply(dEdA2,dAdZ2)*self.learning_rate
		self.b2+=hidden_delta_b

	def draw_sigmoid(self):
		x=[i/10 for i in range(-50,50)]
		y=[self.sigmoid(item) for item in x]
		yp=[self.sigmoid_p(item) for item in x]
		plt.plot(x,y)
		plt.plot(x,yp)
		plt.grid(True)
		plt.show()



brain = NeuralNetwork(2,2,1)

# brain.print_weights()	

print("##")

print(brain.predict([0,0]))
print(sum(brain.error([0,0],[0])))
print("##")

data =     [[[0,0],[0]],
			[[1,1],[0]],
			[[1,0],[1]],
			[[0,1],[1]]]

for i in range(10000):
	item  = random.choice(data)
	brain.backProp(item[0],item[1]) 
print(brain.predict([0,0]))

print(sum(brain.error([0,0],[0])))


while True:
	inp = input(":")
	if(inp=="x"):
		break
	lst = [int(i) for i in inp.split(" ")]

	print(brain.predict(lst[:2]))

# brain.draw_sigmoid()