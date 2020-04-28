import pickle
import os
from collections import Counter
current_directory = os.path.dirname(__file__)
parent_directory = os.path.split(current_directory)[0]
data = pickle.loads(open(os.path.join(parent_directory, 'traindata.pkl'),'rb').read())
print(Counter([d[0] for d in data]).most_common(100))
