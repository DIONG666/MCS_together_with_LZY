import h5py

'''
The data to extract is f['data'][i][0][j][k],
 where `i` is along the time dimension, 
 `j` and `k` are the spatial dimensions.
'''
with h5py.File('../RawData/BJ16_M32x32_T30_InOut.h5', "r") as f:
    extracted_data = f['data'][:, 0, :, :]


output_file_path = '../Predict/data/BJ16_In.h5'
with h5py.File(output_file_path, 'w') as new_file:
    new_file.create_dataset('data', data=extracted_data)
