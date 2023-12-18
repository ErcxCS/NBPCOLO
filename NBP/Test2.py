import numpy as np

data = np.arange(1, 6)
weights = np.random.uniform(0, 1, data.size)
weights /= weights.sum()
print(f"data:{data}\nweights:{weights}")
for i in range(data.size):
    data = np.random.choice(data, size=data.size, replace=True, p=weights)
    
    

print(data)