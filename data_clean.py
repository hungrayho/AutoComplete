#%% imports, config
import pandas as pd
import re
import os

# config
SEP = os.sep
DATA_DIR =  os.path.dirname(os.path.realpath(__file__)) + SEP + "data" + SEP

#%% ingest
df = pd.read_json(DATA_DIR + "Video_Games_5.json", lines=True)

#%% Cleaning
# drop unnecessary columns
df = df[["reviewText"]]

# strip \n
df.replace(r'\n',' ', regex = True, inplace = True) 

# drop missing values
df.dropna(axis = 0, inplace = True)

# split sentences
# df_clean["sentences"] = pd.concat([pd.Series([row['reviewText'].rsplit('.') for _, row in df.iterrows()])]).reset_index()
split = [row['reviewText'].rsplit('.') for _, row in df.iterrows()]

# flatten list
split = [item for sublist in split for item in sublist]

# filter white spaces
split = list(filter(bool, split))

# filter single characters
from itertools import filterfalse

split[:] = filterfalse(lambda x: len(x) == 1, split)

# filter embedded links
import re

# for line in split:
#     line = re.sub(r"(&nbsp)(.*)((<\/a>))", "", line)

split = [re.sub(r"(&nbsp)(.*)((<\/a>))", "", line) for line in split]

# to test:
# list(set(split) - set(re.sub(r"(&nbsp)(.*)((<\/a>))", "", line) for line in split))

# strip leading spaces
split = [line.lstrip() for line in split]

# filter emojis
split = [re.sub(r"( :\)| :D)", "", line) for line in split]


# spell check
# skip
print("cleaning complete...")
#%%
FILE_PATH = DATA_DIR + "dataset.txt"
with open(FILE_PATH, 'w') as f:
    for i in split:
        f.write("%s\n" % i)
print("processed data written to {}".format(FILE_PATH))

# %%
