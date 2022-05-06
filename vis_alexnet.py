import numpy as np

# Paremeters settings
in_path = "./records/alexnet_record.npy"
out_path = "./configs/alexnet"
pen_width = 3
font = "Hilda 10"
color_scheme = 'gnbu9'

# Load neuron records here
records = np.load(in_path, allow_pickle=True)
for i in range(len(records)):
    print(max(records[i]), min(records[i]))

# AlexNet arch:
# conv: 3-64-192-384-256-256
# fc:   256*6*6-4096-4096-100
net_layers = [3, 64, 192, 384, 256, 256]
group_width = 16
step_layer = [1, 2, 3, 4, 5]
layers_label = ['Input']
for i in range(1, len(net_layers)):
    layers_label += [f"Hidden: {net_layers[i]}"]

# Write configuration prefix
def prefix():
    px = ''
    px += "digraph G {" + '\n'
    px += f"\tfontname = \"{font}\"" + '\n'
    px += "\tbgcolor=\"transparent\"" + '\n'
    px += "\trankdir=TB" + '\n'
    px += "\tsplines=line" + '\n'
    px += "\tnodesep=.08;" + '\n'
    px += "\tranksep=1;" + '\n'
    px += "\tedge [color=gray, arrowsize=.3];" + '\n'
    px += "\tnode [fixedsize=true,label=\"\",style=filled," + \
        f"penwidth={pen_width}, color=black,fillcolor=white,shape=circle]\n" + '\n'
    return px

# Write subgraph configuration:
def layer_subgraph(record, step_id, layer_id):
    subgraph = ''
    subgraph += f"\tsubgraph cluster_{layer_id} {{" + '\n'
    subgraph += f"\t\tnode [style=filled, shape=circle, colorscheme={color_scheme}];" + '\n'

    # write nodes
    if layer_id <= step_id and layer_id !=0:
        record = np.sort(record)
        neuron_num = len(record)
        assert neuron_num == net_layers[layer_id]
        mean_record = np.zeros(neuron_num//group_width)
        for i in range(neuron_num//group_width):
            mean_record[i] = record[i*group_width:(i+1)*group_width].mean()
        max_a, min_a = mean_record.max(), mean_record.min()
        for i in range(neuron_num//group_width):
            # r, g, b = color_from_activity(mean_record[i], max_a, min_a)
            # subgraph += f"\t\t l{layer_id}{i} [fillcolor=\"{r} {g} {b}\"] "
            c = color_from_activity(mean_record[i], max_a, min_a)
            subgraph += f"\t\t l{layer_id}{i} [fillcolor={str(c)}] "
            if i != neuron_num -1:
                subgraph += '\n'
    else:
        if layer_id == 0:
            neuron_num = net_layers[layer_id]
            for i in range(neuron_num):
                subgraph += f"\t\t l{layer_id}{i} "
        else:
            neuron_num = len(record)
            assert neuron_num == net_layers[layer_id]
            for i in range(neuron_num//group_width):
                subgraph += f"\t\t l{layer_id}{i} "
    subgraph += ';' + '\n'
    if layer_id == 0:
        subgraph += f"\t\tlabel = \"Input\";" + '\n'
    # subgraph += f"\t\tlabel = \"{layers_label[layer_id]}\";" + '\n'
    subgraph += "\t}\n" + '\n'
    return subgraph

def color_from_activity(activity, m, n):
    c = (activity - n) / (m - n)
    # return c, 0.5, 0.5
    return int(np.round(c*8)+1)

def connections():
    cn = ''
    for i in range(1, len(net_layers)):
        last_layer = net_layers[i-1]//group_width if i!=1 else net_layers[i-1]
        next_layer = net_layers[i]//group_width
        for a in range(last_layer):
            for b in range(next_layer):
                if (b%2 == 0 and a%2 == 0) or i==1:
                    cn += f"\tl{i-1}{a} -> l{i}{b}" + '\n'
    cn += "}" + '\n'
    return cn


# Write Output
for step in step_layer:
    with open(out_path+f"_{step}.dot", 'w') as f:
        f.write(prefix())
        for i in range(len(net_layers)):
            if i == 0:
                f.write(layer_subgraph(None, step, i))
            else:    
                f.write(layer_subgraph(records[i-1], step, i))
        f.write(connections())