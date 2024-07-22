import h5py
import os

def append_to_dataset(h5file, dataset_name, data):
    if dataset_name in h5file:
        dataset = h5file[dataset_name]
        dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
        dataset[-data.shape[0]:] = data
    else:
        maxshape = (None,) + data.shape[1:]
        h5file.create_dataset(dataset_name, data=data, maxshape=maxshape, chunks=True)

def combine_h5_files(input_directory, output_file):
    for filename in os.listdir(input_directory):
        if filename.endswith('.h5'):
            filepath = os.path.join(input_directory, filename)
            with h5py.File(filepath, 'r') as h5file:
                with h5py.File(output_file, 'a') as out_h5file:
                    for name in h5file:
                        data = h5file[name][:]
                        append_to_dataset(out_h5file, name, data)


input_directory = '/mnt/c/Users/User/Documents/AV Research/Val Data/'
output_file = '/mnt/c/Users/User/Documents/AV Research/val_data.h5'
combine_h5_files(input_directory, output_file)
