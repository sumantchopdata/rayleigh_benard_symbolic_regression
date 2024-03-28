#%%
from utils import *
import os

for snapshot_folder in ['snapshots_2e6_1', 'snapshots_2e7_0.5', 'snapshots_2e7_1',
                        'snapshots_2e8_1', 'snapshots_5e7_0.2', 'snapshots_5e7_0.03']:
    all_snapshots = sorted(os.listdir(snapshot_folder))
    my_snapshot = read_snapshots(snapshot_folder + '/' + all_snapshots[-1])
    
    print(snapshot_folder.split('_')[-2:], all_snapshots[-1])
    # print if my_snapshot[0].values() is all Nan or not
    for i in range(len(my_snapshot)):
        print(list(my_snapshot[i].keys())[0], np.isnan(list(my_snapshot[i].values())).all())
    print()
# %%
