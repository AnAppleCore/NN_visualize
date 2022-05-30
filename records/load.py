import numpy as np

# a = np.load('alexmnist_gaba45_0_active_feaature.npy', allow_pickle=True)
# for i in range(len(a)):
#     print(a[i].shape)
#     if i ==0:
#         a[i] = np.mean(a[i], axis=(1,2))
#     print(a[i].shape)
# np.save('alexnet_gaba_record.npy', a, allow_pickle=True)
# a = np.load('alexnet_gaba_record.npy', allow_pickle=True)
# for i in range(len(a)):
#     print(a[i].shape)

for j in range(20):
    print(f"=>{j}")
    a = np.load(f'./gaba/alexnet_mnist_gaba45_epoch:{j}__0_active_feature.npy', allow_pickle=True)
    for i in range(len(a)):
        print(f"before: {a[i].shape}")
        if i == 0:
            a[i] = np.mean(a[i], axis=(1,2))
        print(f"after: {a[i].shape}")
    np.save(f'./gaba/gaba_epoch{j}.npy', a, allow_pickle=True)

    print("\nreload test")
    a = np.load(f'./gaba/gaba_epoch{j}.npy', allow_pickle=True)
    for i in range(len(a)):
        print(a[i].shape)