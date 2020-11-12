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
        self.learning_rate=0.2


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
            dEdA =      errors[i]
            dAdZ =      np.multiply(layers_tensors_r[i],(1-layers_tensors_r[i])) # sigmoid_prime (derivative)
            dZdW =      layers_tensors_r[i+1]

            #         changing second weights
            #         (------------dot product------) * learning_rate  
            #         (---cross product-----) |       |
            delta =  (np.multiply(dEdA,dAdZ) * dZdW.T) * self.learning_rate
            self.weights[n-1-i] += delta

            #         changing second biases
            #         (---cross product-----)    * learning_rate 
            delta_b =  (np.multiply(dEdA,dAdZ)) * self.learning_rate
            self.biases[n-1-i] += delta_b
        return


    def train(self,x,y,test=None,iteration=False,lr_r=0.95):
        n = len(x)
        x=np.array(x)
        y=np.array(y)
        if not iteration:
            iteration=len(x)
        for i in range(iteration):
            index = random.randint(0,n-1)
            self.backProp(x[index],y[index])
            # if i%1000==0:
            #   self.learning_rate*=lr_r
        if test:
            test_x,test_y = test
            iteration=len(x)
            cnt=0
            for i in range(iteration):
                index = random.randint(0,n-1)
                arr=self.predict(x[index],y[index])
                if test_y.index(1)==arr.index(1):
                    cnt+=1
                if i%100:
                    print(f":{i}")
            print(f"accuracy {cnt/iteration *100}%")