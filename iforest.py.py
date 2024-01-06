import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

# External Node class for leaf nodes in the Isolation Tree
class exNode:
    def __init__(self, size):
        self.size = size

# Internal Node class for non-leaf nodes in the Isolation Tree
class inNode:
    def __init__(self, left, right, normal, intercept):
        self.left = left
        self.right = right
        self.normal = normal
        self.intercept = intercept
        
# Function to build an Isolation Forest
def iForest(X, t, psi):
    forest = []
    """" 
    height_limit is in fact the average height of the tree that would be
    constructed from given points. This acts as height_limit for the
    construction because we are only interested in data points that have
    shorter-than-average path lengths, as those points are more likely
    to be anomalies.
    
    """
    height_limit = int(np.ceil(np.log2(psi)))

    for _ in range(t):
        X_prime = sample(X, psi)
        forest.append(iTree(X_prime, 0, height_limit))

    return forest

# Recursive function to build an Isolation Tree
def iTree(X, e, l):
    """
  The function constructs a tree/sub-tree on points X.

  e: represents the height of the current tree to
    the root of the decision tree.
  l: the max height of the tree that should be constructed.

  The e and l only exists to make the algorithm efficient
  as we assume that no anomalies exist at depth >= l.
  """
    if e >= l or len(X) <= 1:
        return exNode(len(X))
    else:
        normal = np.random.normal(size=X.shape[1])
        intercept = np.random.uniform(X.min(axis=0), X.max(axis=0))
        normal[np.random.choice(X.shape[1], int(np.ceil(X.shape[1] / 2)), replace=False)] = 0

        X_left = X[(X - intercept).dot(normal) <= 0]
        X_right = X[(X - intercept).dot(normal) > 0]

        return inNode(iTree(X_left, e + 1, l), iTree(X_right, e + 1, l), normal, intercept)
    
# Function to compute the path length of an instance in an Isolation Tree
def path_length(x, tree, e=0):
    """
      The function returns the path length of an instance
      x in tree `T`. Path Length of a point x is the number 
      of edges x traverses from the root node.
      here e is the number of edges traversed from the root till the current
      subtree T.
  
    """
    if isinstance(tree, exNode):
        return e + c(tree.size)

    normal = tree.normal
    intercept = tree.intercept

    if (x - intercept).dot(normal) <= 0:
        return path_length(x, tree.left, e + 1)
    else:
        return path_length(x, tree.right, e + 1)
    
# Function to compute the average path length for a given node size
def c(size):
    """"
    c(n) is the average of path length given n, 
    we use it to normalize path_length.
    
    """
    if size > 2:
        return 2 * (np.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size)
    elif size == 2:
        return 1
    else:
        return 0

# Function to randomly sample instances from the dataset
def sample(X, psi):
    indices = np.random.choice(len(X), psi, replace=False)
    return X[indices, :]

# Function to simulate a data stream with anomalies
def simulate_data_stream():
    mu = 0
    sigma = 1
    stream_size = 3000
    anomalies = [800, 801, 802, 803]  # Indices of simulated anomalies

    data_stream = np.random.normal(mu, sigma, size=(stream_size, 2))

    for anomaly in anomalies:
        data_stream[anomaly] = np.random.normal(mu + 10, sigma, size=(1, 2))

    return data_stream
    
# Function to visualize 'anomalies' in the 'data stream'  
def visualize_anomalies(data_stream, anomalies, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('Reds')
    ax.set_facecolor('#7B0000')  # Dark Orange background color
    heatmap = ax.hexbin(data_stream[:, 0], data_stream[:, 1], gridsize=50, cmap=cmap, norm=LogNorm(), mincnt=1)

    ax.scatter(data_stream[anomalies, 0], data_stream[anomalies, 1], c='#FFA07A', marker='o', s=20, label='Detected Anomalies')

    ax.set_title(title)
    ax.legend()

    plt.show()

def main():
    np.random.seed(42)
    data_stream = simulate_data_stream()

    t = 100  # Number of trees
    psi = 256  # Sub-sampling size

    # Train the iForest
    forest = iForest(data_stream, t, psi)

    detected_anomalies = []

    # For each instance in the data stream, calculate the path lengths to all trees in the Isolation Forest.
    for i, instance in enumerate(data_stream):
        
        # Compute the path lengths for the current instance across all trees in the forest.
        path_lengths = [path_length(instance, tree) for tree in forest]
        
        # Calculate the average path length for the current instance.
        avg_path_length = np.mean(path_lengths)
        threshold = c(psi)

        # Check if the average path length exceeds the threshold, indicating a potential anomaly.
        if avg_path_length > threshold:
            detected_anomalies.append(i)

    
    visualize_anomalies(data_stream, detected_anomalies, title='Anomalies Detected Testing')

if __name__ == "__main__":
    main()
