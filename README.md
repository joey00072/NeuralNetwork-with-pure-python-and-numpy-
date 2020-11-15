# NeuralNetwork-with-pure-python-and-numpy

Classic Neural Network implimentation in pyhton with numpy

```python
from neural_network import NeuralNetwork

layers = [2,4,4,1]
brain = NeuralNetwork(layers)

#Sample data for XOR
train_x = [[0, 0], [1, 1], [1, 0], [0, 1]]
train_y = [[0], [0], [1], [1]]

brain.train(train_x,train_y)

pred = brain.predict([1,0])

print(pred)
```

---

# XOR problem Visualtion

Pygame required

run

```
$ python3 xor.py
```

<img src="https://camo.githubusercontent.com/..." data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="200" height="400" />

![](images/xor_soln1_new.PNG = 250x)
![](images/xor_soln2_new.PNG =250x250)
Neural Network reaching same conclusion with different solution space  
(corners for both have same color)
