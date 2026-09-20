# check_h5.py
import h5py
import sys

def print_hdf5_structure(name, obj):
    """Prints the name and type of h5py objects."""
    # Indent based on the depth of the group
    depth = name.count('/')
    indent = '    ' * depth
    
    if isinstance(obj, h5py.Group):
        print(f"{indent}📂 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_h5.py /path/to/your/file.h5")
        sys.exit(1)
        
    h5_path = sys.argv[1]
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"Inspecting structure of: {h5_path}\n")
            f.visititems(print_hdf5_structure)
    except Exception as e:
        print(f"Error opening or reading HDF5 file: {e}")