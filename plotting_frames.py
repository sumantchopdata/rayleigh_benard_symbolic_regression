# plot snapshots for all the files in the folder
import os

file_list = sorted(os.listdir('snapshots'))

for file in file_list:
    os.system(f'python3 plot_snapshots.py {file}')