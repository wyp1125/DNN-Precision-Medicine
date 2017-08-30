#This program divides a dataset into train and testing sets.

from __future__ import print_function
import sys
import random
if len(sys.argv)<4:
  print("Usage:python divide_train_test.py input output_prefix train-test_fold")
  quit()
with open(sys.argv[1]) as f:
  content=[x.strip('\n') for x in f.readlines()]
header=content[0]
m=len(content)-1
k=int(sys.argv[3])+1
n=int(round(m/k))
id=list(range(m))
sel=random.sample(id,n)
with open(sys.argv[2]+'.train.csv','w') as out1:
  out1.write(header+'\n')
  with open(sys.argv[2]+'.test.csv','w') as out2:
    out2.write(header+'\n')
    for i in range(m):
      if i in sel:
        out2.write(content[i+1]+'\n')
      else:
        out1.write(content[i+1]+'\n')


