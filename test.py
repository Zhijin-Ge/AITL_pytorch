import random
import numpy as np
for i in range(100):
    transform_index = list(random.randint(1, 20) for _ in range(4))
    print(transform_index)
    number_strings = [str(num) for num in transform_index]
    trans_lists = "_".join(number_strings)
    print(trans_lists)