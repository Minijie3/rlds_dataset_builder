import h5py

def print_hdf5_structure(group, indent=0):
    for key in group.keys():
        item = group[key]
        line = ' ' * indent + f'↳ {key}'
        if isinstance(item, h5py.Group):
            print(line + ' (Group)')
            print_hdf5_structure(item, indent + 4)
        elif isinstance(item, h5py.Dataset):
            info = f' (Dataset) - Shape: {item.shape}, Dtype: {item.dtype}'
            print(line + info)

def explore_hdf5(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"\nFile path: {filepath}")
            print('=' * 50)
            print_hdf5_structure(f)
            print('=' * 50)

            print(len(f['data']))
            print(f['data'].keys())
    except Exception as e:
        print(f"[ERROR] File not found: {str(e)}")
    

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        filepath = input("[Input] Path of .hdf5 file: ")
    else:
        filepath = sys.argv[1]
    
    explore_hdf5(filepath)