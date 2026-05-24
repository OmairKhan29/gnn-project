import os
import ctypes
import ctypes.util

# Add all dll directories first
import os
for d in [
    r'C:\Users\khano\.conda\envs\gnn-env-new',
    r'C:\Users\khano\.conda\envs\gnn-env-new\Library\bin',
]:
    if os.path.exists(d):
        os.add_dll_directory(d)

# Try loading rdBase.pyd directly and catch the error
rdbase_path = r'C:\Users\khano\.conda\envs\gnn-env-new\Lib\site-packages\rdkit\rdBase.pyd'

print(f"Trying to load: {rdbase_path}")
print(f"File exists: {os.path.exists(rdbase_path)}")

try:
    lib = ctypes.CDLL(rdbase_path)
    print("Loaded successfully!")
except OSError as e:
    print(f"Failed: {e}")

# Check what DLLs are in the env root
print("\nDLLs in env root:")
env_root = r'C:\Users\khano\.conda\envs\gnn-env-new'
dlls = [f for f in os.listdir(env_root) if f.endswith('.dll') or f.endswith('.pyd')]
for d in sorted(dlls)[:20]:
    print(f"  {d}")

# Check Library\mingw-w64\bin
mingw = r'C:\Users\khano\.conda\envs\gnn-env-new\Library\mingw-w64\bin'
if os.path.exists(mingw):
    print(f"\nDLLs in mingw bin (first 10):")
    for f in sorted(os.listdir(mingw))[:10]:
        print(f"  {f}")
else:
    print(f"\nNo mingw-w64 folder found")
