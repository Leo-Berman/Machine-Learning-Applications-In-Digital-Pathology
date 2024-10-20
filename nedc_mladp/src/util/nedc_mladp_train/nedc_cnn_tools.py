import numpy as np
import random

def randomData(num_frames:int):

    # Set the random seed for reproducibility
    np.random.seed(42)

    # Create a NumPy array of random 32x32x4 arrays
    # data = np.random.rand(num_frames, 32, 32, 4)

    # Create a NumPy array of random 1x1000 arrays
    data = np.random.rand(num_frames, 1, 1000)

    # Generate random label digits (as strings) corresponding to the number of arrays
    labels = np.array([str(np.random.randint(0, 10)) for _ in range(num_frames)])

    return data, labels

def parsingFeats():
    pass