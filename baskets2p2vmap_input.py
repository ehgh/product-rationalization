import os
import numpy as np
from timeit import default_timer as timer
from os.path import join as join_path


data_direcotry = 'data_graph'
#out_directory = join_path('..','p2v-map','data')
out_directory = join_path('p2v')

file_list = ['baskets_test.csv', 'baskets_train.csv', 'baskets_validation.csv']
#files to get p2v embedding
open(join_path('p2v', 'baskets.csv'), 'w').close()
open(join_path('p2v', 'baskets_train.csv'), 'w').close()
open(join_path('p2v', 'baskets_test.csv'), 'w').close()
open(join_path('p2v', 'baskets_validation.csv'), 'w').close()

#open files to write output to and add header line
for filename in file_list:
  with open(join_path(out_directory, filename), 'w') as f_w:
    basket_hash = 0
    f_w.write(',i,j,price,price_paid,discount,t,basket_hash\n')
    with open(join_path(data_direcotry, filename), 'r') as f_r:
      next(f_r)
      for line in f_r:
        line = list(set(line.strip().split(',')))
        for prod in line:
          f_w.write(',0,'+prod+',1,1,0,0,'+str(basket_hash)+'\n')
        basket_hash += 1
