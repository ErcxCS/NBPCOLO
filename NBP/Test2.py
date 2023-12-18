from scipy.stats import gaussian_kde
import numpy as np
from matplotlib import pyplot as plt

""" x = np.random.normal(20, 0.01, 100) # generate 100 random values from a normal distribution with mean 0 and standard deviation 0.01
count, bins, patches = plt.hist(x, bins=10, density=True, alpha=0.5, label='Histogram') # plot the histogram of the data with 20 bins, density=True, and 50% transparency
plt.xlabel('X') # add x-axis label
plt.ylabel('Probability Density') # add y-axis label
plt.legend() # add legend
print(count, bins)
weights = np.random.uniform(0, 1, bins.shape)
weights /= weights.sum()
weights[-1] = 55
kde = gaussian_kde(bins, weights=weights) # create a KDE object from the data
x_eval = np.linspace(20-0.03, 20+0.03, 100) # create an array of values from -0.03 to 0.03 with 0.1 increments
evaluated = kde(x_eval) # evaluate the KDE at the x_eval values
print(weights)
plt.plot(x_eval, evaluated, label='KDE') # plot the KDE with the x_eval values on the x-axis and the evaluated values on the y-axis
plt.show() # display the plot """


""" Z = np.reshape(m, X.shape)
plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[x_min, x_max, y_min, y_max])
plt.scatter(test_points[:, 0], test_points[:, 1])
plt.scatter(test_points[np.argmax(m), 0], test_points[np.argmax(m),  1])
plt.show() """

def measure(n):
    #"Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(size=n)
    return m1+m2, m1-m2
m1, m2 = measure(2000)
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()
#Perform a kernel density estimate on the data:

X, Y = np.mgrid[xmin:xmax:10j, ymin:ymax:10j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])

kernel =gaussian_kde(values)

Z = np.reshape(kernel(positions), X.shape)
print(Z.max())
#Plot the results:
print(
    f"pos: {positions.shape}\nZ: {Z.shape}\nval:{values.shape}"
)

fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax])
ax.plot(m1, m2, 'k.', markersize=3)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()
