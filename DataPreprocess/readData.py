import h5py

with h5py.File('../Data/BJ16In.h5', "r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
