import numpy as np
import pandas as pd
import random
import os

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

def spilt_pd(train_df,pd_list,name):
    col = ['Image','Id']
    val_df = pd.DataFrame([],columns=col)
    # val_df['Id'] = ''
    for i in range(len(pd_list)):
        val_df.loc[i, 'Image'] = train_df.iloc[pd_list[i]]['Image']
        val_df.loc[i, 'Id'] = train_df.iloc[pd_list[i]]['Id']
    val_df.to_csv(name, index=False)
    print(val_df.head(5))

os.listdir("../input/")

train_df = pd.read_csv("../input/train.csv")
train_df.head()


# lets find total number of different whales present
print('Training examples: {len(df)}')
print("Unique whales: ",train_df['Id'].nunique()) # it includes new_whale as a separate type.

training_pts_per_class = train_df.groupby('Id').size()
training_groups = train_df.groupby('Id').groups
print("Min example a class can have: ",training_pts_per_class.min())
print("Max example a class can have: \n",training_pts_per_class.nlargest(200))
print("Max example index: \n",training_pts_per_class.nlargest(5005).index[200:])

val_list=[]
train_list = []

train_class = training_pts_per_class.nlargest(802)
print("Max example index v: \n",train_class.index[1])
print("Max example index v1: \n",training_groups[train_class.index[1]][0])

# gen val_list
for i in range(200):
    val_list.append(training_groups[train_class.index[0]][i])

for i in range(800):
    val_list.append(training_groups[train_class.index[i+1]][0])

print(len(val_list))
print(val_list)
random.shuffle(val_list)
print(len(val_list))
print(val_list)

# gen train_list
for i in range(25360):
    train_list.append(i)
for i in range(999):
    train_list.remove(val_list[i])



print(len(train_list))
print(train_list)

#print("Max example index v2: \n",train_df.iloc[val_list[0]])



# spilt_pd(train_df,val_list,'val.csv')
# spilt_pd(train_df,train_list,'train_s.csv')

