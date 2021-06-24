#%% imports, config
import pandas as pd
import re
import os

from itertools import filterfalse

# config
SEP = os.sep
DATA_DIR =  os.path.dirname(os.path.realpath(__file__)) + SEP + "data" + SEP

#%% ingest
df = pd.read_json(DATA_DIR + "Video_Games_5.json", lines=True)

#%% Cleaning json payload in pandas
# drop unnecessary columns
df = df[["reviewText"]]

# strip \n
df.replace(r'\n',' ', regex = True, inplace = True) 

# drop missing values
df.dropna(axis = 0, inplace = True)

print("extracted data from json payload")
#%% cleaning extracted text in python
# split sentences around "."
# df_clean["sentences"] = pd.concat([pd.Series([row['reviewText'].rsplit('.') for _, row in df.iterrows()])]).reset_index()
split = [row['reviewText'].rsplit('.') for _, row in df.iterrows()]

# split sentences around "!"
# for sublist in split:
#     for line in sublist:
split = [line.rsplit('!') for sublist in split for line in sublist]

# split sentences around "?"
# for sublist in split:
    # for line in sublist:
split = [line.rsplit('?') for sublist in split for line in sublist]

# flatten list
# for sublist in split:
#     for line in sublist:
split = [line for sublist in split for line in sublist]

# filter empty strings
split = list(filter(bool, split))

# filter single characters
split[:] = filterfalse(lambda x: len(x) == 1, split)

# filter single words
regex = re.compile("^[A-Za-z]+$")
split[:] = filterfalse(regex.match, split)

# filter embedded links
# for line in split:
#     line = re.sub(r"(&nbsp)(.*)((<\/a>))", "", line)
split = [re.sub(r"(&nbsp)(.*)((<\/a>))", "", line) for line in split]
# to test:
# list(set(split) - set(re.sub(r"(&nbsp)(.*)((<\/a>))", "", line) for line in split))

# strip leading spaces
split = [line.lstrip() for line in split]

# filter emoticons
split = [re.sub(r"( :\)| :D)", "", line) for line in split]

# TODO: spell check

print("data cleaning complete...")
#%%
FILE_PATH = DATA_DIR + "dataset.txt"
with open(FILE_PATH, 'w') as f:
    for i in split:
        f.write("%s\n" % i)
print("processed data written to {}".format(FILE_PATH))
