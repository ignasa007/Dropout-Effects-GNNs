import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dirs', nargs='+', type=str)
args = parser.parse_args()

for dir in args.dirs:

    if dir in ('__pycache__', 'overleaf'):
        continue

    if not os.path.isdir(dir):
        print(f'Path {dir} not found.')
        continue

    src_names = [
        f'{dirpath}/{f}'.replace('\\', '/')
        for (dirpath, _, filenames) in os.walk(dir) 
        for f in filenames
    ]

    dst_names = [
        f"overleaf/{f.replace(' ', '-').replace('_', '-').replace('/', '_')}"
        for f in src_names
    ]

    for src_name, dst_name in zip(src_names, dst_names):
        
        if os.path.isfile(dst_name):
            print(dst_name)
            os.remove(dst_name)
        os.makedirs(os.path.dirname(dst_name), exist_ok=True)
        shutil.copy2(src_name, dst_name)