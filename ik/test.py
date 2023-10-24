import pickle as pk
import numpy as np

import joblib



with open('ik/res.pk', 'rb') as f:
    data = pk.load(f)['pred_xyz_24_struct']

print("Hbrik")
print(np.shape(data))
# print(data[0])

c_data = data.copy()

n_data = c_data *2.1

with open('ik/res_v2.pk', 'wb') as f:
    pk.dump(n_data, f)

# print(n_data[5])


# file_path = 'ik/vibe_output.pkl'
# data = joblib.load(file_path)[1]['joints3d']

# print("Vibe")
# print(np.shape(data))
# print(data[0])

