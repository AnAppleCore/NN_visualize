import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Paremeters settings
in_path = "./records/alexnet_gaba_record.npy"
out_path = "./img/curve_gaba_color_bar"

records = np.load(in_path, allow_pickle=True)
sorted_records = []
for i in range(len(records)):
    sorted_records.append(np.sort(records[i])[::-1])

# Normalization:
for i in range(len(records)):
    max, min = sorted_records[i].max(), sorted_records[i].min()
    sorted_records[i] = (sorted_records[i] - min) / (max - min)

for i in tqdm.trange(len(sorted_records), ncols=80):
    neuron_num = len(sorted_records[i])
    neurons = np.arange(neuron_num)

    # transparent background
    fig = plt.figure(i, figsize=(neuron_num, 10))
    fig.patch.set_alpha(0.0)
    ax = plt.gca()
    ax.patch.set_alpha(0.0)

    # plot
    # plt.plot(neurons, sorted_records[i], color='b',marker=',', linestyle='solid', linewidth=30.0)

    # bar
    # plt.bar(neurons, sorted_records[i], width=1, edgecolor='white', linewidth=0.7)

    # color bar
    color_map = cm.get_cmap(name='GnBu')
    color = color_map(sorted_records[i])
    plt.bar(neurons, sorted_records[i], width=1, color=color)

    # stem
    # plt.stem(neurons, sorted_records[i])

    plt.savefig(out_path+f'_{i}.png')
