import h5py

'''
The data to extract is f['data'][i][0][j][k],
 where `i` is along the time dimension, 
 `j` and `k` are the spatial dimensions.
'''
extracted_data = None
with h5py.File('../Data/BJ16_M32x32_T30_InOut.h5', "r") as f:
    # Extract the desired data: data[i][0][j][k] for all i, j, k
    extracted_data = f['data'][:, 0, :, :]

output_file_path = '../Data/BJ16In.h5'
with h5py.File(output_file_path, 'w') as new_file:
    new_file.create_dataset('BJ16_In', data=extracted_data)
