import os
import pickle

values = {'entry':[]}
file = 'check.pkl'
file_path = os.path.join(os.getcwd(),file)

for i in range(1,50):
    values['entry'].append(i)
    pickle.dump(values, open(file_path, 'wb'))
for i in range(50,100):
    values['entry'].append(i)
    pickle.dump(values, open(file_path, 'wb'))

print("Done")


