import numpy as np

a = np.load('alexmnist_gaba45_0_active_feaature.npy', allow_pickle=True)
for i in range(len(a)):
    print(a[i].shape)
    if i ==0:
        a[i] = np.mean(a[i], axis=(1,2))
    print(a[i].shape)
np.save('alexnet_gaba_record.npy', a, allow_pickle=True)
a = np.load('alexnet_gaba_record.npy', allow_pickle=True)
for i in range(len(a)):
    print(a[i].shape)