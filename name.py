import matplotlib.pyplot as plt
import random

arr=[0]*10
for i in range(1000):
	n=random.randint(0,10-1)
	arr[n]+=1

print(arr)
plt.hist(arr,bins=5)
plt.show()
