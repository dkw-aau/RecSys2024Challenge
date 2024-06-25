from argparse import ArgumentParser
import pickle 
import numpy as np
from tqdm.auto import tqdm
import zipfile
import io


# In[3]:
parser = ArgumentParser()
parser.add_argument('--exp', type=str)
args = parser.parse_args()
exp = args.exp

data_path = f'inference_{exp}.pickle' 
zip_file = f'3-model_{exp}.zip' 

print(f'Generation results for experiment {exp}')


with open(data_path, 'rb') as f:
    data = pickle.load(f)


# In[5]:


# with open('predictions.txt', 'w') as f:
#     for d in tqdm(data):
#         f.write(f'{d[0]} [{",".join(map(str,d[1]))}]')
#         f.write('\n')


# In[6]:


with zipfile.ZipFile(zip_file, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    with io.TextIOWrapper(z.open('predictions.txt', 'w')) as f:
        for d in tqdm(data):
            f.write(f'{d[0]} [{",".join(map(str,d[1]))}]')
            f.write('\n')


# In[7]:


print('Done!')


# In[ ]:




