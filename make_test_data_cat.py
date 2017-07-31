#This program makes a simulated dataset for DNN application programms.
#The last several (no more than 10) columns before 'label' contain categorical values.

from __future__ import print_function
import numpy as np
import sys

if len(sys.argv)<5:
  print("Usage:python make_test_data_cat.py file #class #samples #features #categorical")
  quit()

ss=['a','b','c','d','e','f','g','h','i','j']

with open(sys.argv[1],'w') as fl:
  for i in range(int(sys.argv[4])):
    fl.write('s'+str(i)+',')
  fl.write('label\n')
  for i in range(int(sys.argv[3])):
    mu=int(np.random.uniform(0,int(sys.argv[2])))
    s=np.random.normal(mu,1,int(sys.argv[4])-int(sys.argv[5]))
    for j in s:
      k=int((j+0.05)*100)/100
      fl.write(str(k)+',')
    for j in range(int(sys.argv[5])):
      t=ss[int(np.random.uniform(0,int(sys.argv[2])))]
      fl.write(str(t)+',')
    fl.write(str(mu)+'\n')


