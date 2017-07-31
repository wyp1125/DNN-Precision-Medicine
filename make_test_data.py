#This program makes a simulated dataset for DNN application programms.
#All the feature data are real values

from __future__ import print_function
import numpy as np
import sys

if len(sys.argv)<4:
  print("Usage:python make_test_data.py file #class #samples #features")
  quit()

with open(sys.argv[1],'w') as fl:
  fl.write(str(sys.argv[3])+','+str(sys.argv[4])+','+str(sys.argv[2])+'\n')
  for i in range(int(sys.argv[3])):
    mu=int(np.random.uniform(0,int(sys.argv[2])))
    s=np.random.normal(mu,1,int(sys.argv[4]))
    for j in s:
      k=int((j+0.05)*100)/100
      fl.write(str(k)+',')
    fl.write(str(mu)+'\n')


