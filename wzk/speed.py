import time
import numpy as np


def test_computer_speed():
    start_time = time.time()

    for i in range(100):
        # Perform some dummy calculations using NumPy
        a = np.random.rand(1000, 1000)
        b = np.random.rand(1000, 1000)
        result = np.dot(a, b)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed Time: {elapsed_time} seconds")


test_computer_speed()

# 0062 3.5