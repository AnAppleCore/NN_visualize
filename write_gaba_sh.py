import os

with open('gabash.txt', 'w') as f:
    for i in range(20):
        f.write('python vis_alexnet.py \\\n')
        f.write(f'-i ./records/gaba/gaba_epoch{i}.npy \\\n')
        f.write(f'-o ./configs/gaba/gaba_epoch{i} \\\n')
        f.write('\n')
        f.write(f'dot -Tpng -o ./img/gaba/gaba_epoch{i}_5.png ./configs/gaba/gaba_epoch{i}_5.dot\n')
        f.write(f'dot -Tpdf -o ./img/gaba/gaba_epoch{i}_5.pdf ./configs/gaba/gaba_epoch{i}_5.dot\n')
        f.write('\n')
