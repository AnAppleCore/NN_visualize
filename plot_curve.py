import numpy as np
import matplotlib.pyplot as plt

# Paremeters settings
in_path = "./records/alexnet_record.npy"
out_path = "./img/curve"

records = np.load(in_path, allow_pickle=True)
sorted_records = []
for i in range(len(records)):
    sorted_records.append(np.sort(records[i])[::-1])
print(len(sorted_records))