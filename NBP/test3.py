import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a list of line segments
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a set of varying line widths
widths = np.linspace(0.1, 5, len(x) - 1)

# Create a LineCollection with varying widths
lc = LineCollection(segments, linewidths=widths, cmap='viridis')

# Create the plot
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Plot with Varying Line Widths')
plt.show()