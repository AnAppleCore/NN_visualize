import numpy as np

# Paremeters settings
in_path = "./alexnet_record.npy"
out_path = "alexnet.txt"
pen_width = 5
font = "Hilda 10"

# Load neuron records here
records = np.load(in_path, allow_pickle=True)
for i in range(len(records)):
    print(max(records[i]), min(records[i]))

# AlexNet arch:
# conv: 3-64-192-384-256-256
# fc:   256*6*6-4096-4096-100
net_layers = [3, 64, 192, 384, 256, 256]
layers_label = ['Input']
layers_col = ['none']
layers_fill = ["black"]
for i in range(1, len(net_layers)):
    layers_label += [f"Hidden: {net_layers[i]}"]
    layers_col += ['none']
    layers_fill += ["gray"]

# Write configuration prefix
def prefix():
    px = ''
    px += "digraph G {" + '\n'
    px += f"\tfontname = \"{font}\"" + '\n'
    px += "\trankdir=LR" + '\n'
    px += "\tsplines=line" + '\n'
    px += "\tnodesep=.08;" + '\n'
    px += "\tranksep=1;" + '\n'
    px += "\tedge [color=black, arrowsize=.5];" + '\n'
    px += "\tnode [fixedsize=true,label=\"\",style=filled," + \
        "color=none,fillcolor=gray,shape=circle]\n" + '\n'
    return px

# Write subgraph configuration:
def layer_subgraph(record, layer_id):
    subgraph = ''
    subgraph += f"\tsubgraph cluster_{layer_id} {{" + '\n'
    subgraph += f"\t\tcolor={layers_col[layer_id]};" + '\n'
    subgraph += f"\t\tnode [style=filled, color=white, penwidth={pen_width}, fillcolor={layers_fill[layer_id]}, shape=circle];" + '\n'

    # write nodes
    if layer_id != 0: # hidden layer
        record = np.sort(record)[::-1]
        neuron_num = len(record)
        assert neuron_num == net_layers[layer_id]
        max_a, min_a = record.max(), record.min()
        for i in range(neuron_num):
            r, g, b = color_from_activity(record[i], max_a, min_a)
            subgraph += f"\t\t l{layer_id}{i} [fillcolor=\"{r} {g} {b}\"] "
            if i != neuron_num -1:
                subgraph += '\n'
    else: # input layers
        neuron_num = 3
        assert neuron_num == net_layers[layer_id]
        for i in range(3):
            subgraph += f"\t\t l{layer_id}{i} "
    subgraph += ';' + '\n'
    subgraph += f"\t\tlabel = \"{layers_label[layer_id]}\";" + '\n'
    subgraph += "\t}\n" + '\n'
    return subgraph

def color_from_activity(activity, m, n):
    # c = 1.0/(1.0+np.exp(-activity*20)) # sigmoid
    c = (activity - n) / (m - n)
    return c, c, c

def connections():
    cn = ''
    for i in range(1, len(net_layers)):
        for a in range(net_layers[i-1]):
            for b in range(net_layers[i]):
                if b % (net_layers[i]/4) == 0 and a % (net_layers[i-1]/4) == 0:
                # if b % (net_layers[i]/16) == 0 and a % (net_layers[i-1]/16) == 0:
                # if a == b:
                    cn += f"\tl{i-1}{a} -> l{i}{b}" + '\n'
    cn += "}" + '\n'
    return cn

with open(out_path, 'w') as f:
    f.write(prefix())
    for i in range(len(net_layers)):
        if i == 0:
            f.write(layer_subgraph(None, i))
        else:    
            f.write(layer_subgraph(records[i-1], i))
    f.write(connections())