import sys
import pickle
import re

all_data = {}
for line in sys.stdin:
    l = line.split("\t")
    l[2] = re.sub(r'\([^)]*\)', '', l[2])
    if l[1] in all_data:
        all_data[l[1]].append(l[2].strip())
    else:
        all_data[l[1]]=[l[2]]

with open("test_res.pickle","wb") as f:
    pickle.dump(all_data,f)
