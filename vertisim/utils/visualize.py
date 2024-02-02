import numpy as np

class Visualizer:
    """Visualize data."""
    
    def __init__(self, plt):
        self.plt = plt
    
    def visualize_vectors(self, **kwargs) -> None:
        self.plt.figure()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        origin = np.array([0, 0])
        
        max_x, max_y = -np.inf, -np.inf
        min_x, min_y = np.inf, np.inf
        
        for i, (label, v) in enumerate(kwargs.items()):
            if label=='Title':
                self.plt.title(v, fontsize=22)
            else:
                color = colors[i % len(colors)]
                self.plt.quiver(*origin, *v, color=color, angles='xy', scale_units='xy', scale=1, label=label)
                max_x, max_y = max(max_x, v[0]), max(max_y, v[1])
                min_x, min_y = min(min_x, v[0]), min(min_y, v[1])
    
        limit = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y)) + 1
        self.plt.xlim(-limit, limit)
        self.plt.ylim(-limit, limit)
        self.plt.axhline(0, color='black',linewidth=0.5)
        self.plt.axvline(0, color='black',linewidth=0.5)
        self.plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        self.plt.legend(fontsize=18)
    
    def show(self):
        self.plt.show()
    