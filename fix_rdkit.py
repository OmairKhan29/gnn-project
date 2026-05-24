import os
import ctypes
import sys

# Add all possible DLL directories
dirs_to_add = [
    r'C:\Users\khano\.conda\envs\gnn-env-new',
    r'C:\Users\khano\.conda\envs\gnn-env-new\Library\bin',
    r'C:\Users\khano\.conda\envs\gnn-env-new\Library\mingw-w64\bin',
    r'C:\Users\khano\.conda\envs\gnn-env-new\Library\usr\bin',
]

for d in dirs_to_add:
    if os.path.exists(d):
        os.add_dll_directory(d)
        print(f"Added: {d}")

print("\nNow trying rdkit import...")
try:
    from rdkit import Chem
    print("RDKit OK!")
except ImportError as e:
    print(f"Failed: {e}")
