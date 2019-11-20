import os
import argparse
import numpy as np
from timeit import default_timer as timer
from os.path import join as join_path

np.set_printoptions(precision = 1, suppress = True)


def data_generator(args):
  
  #create data directory and delete content of files if already exist
  output_directory = 'basket_completion_data'
  if not os.path.isdir(output_directory):
    os.mkdir(output_directory)
  open(join_path(output_directory, args.output + '_test.csv'), 'w').close()
  
  #open files to write output to and add header line
  data_directory = 'data'
  basket_completion_file = open(join_path(output_directory, args.output + '_test.csv'), 'a')
  baskets_file_train = open(join_path(data_directory, args.output + '_train.csv'), 'r')
  baskets_file_test = open(join_path(data_directory, args.output + '_test.csv'), 'r')
  

  start = timer()
  #basket generation
  weights = 
  cnt = -1
  train_cnt = -1
  test_cnt = -1
  val_cnt = -1
  basket_id = -1

  data_split = list(map(float, args.data_split.split(',')))

  for i in range(args.I):
#    baskets_i = []
    for t in range(args.T):
      basket = []
      basket_id += 1
      for category in range(args.C):
        if y[i, t, category]:
          cnt += 1
          idx = np.argmax(utility_c_ijt[category, i, :, t])
          j = category * args.Jc + idx
          basket.append(j)
          p2v_basket_file.write(','.join(map(str, 
            [cnt, i, j, 1, 1, 0, t, basket_id])) + '\n')
          if float(t)/args.T < 0.8:
            train_cnt += 1
            p2v_basket_file_train.write(','.join(map(str, 
              [train_cnt, i, j, 1, 1, 0, t, basket_id])) + '\n')
          elif float(t)/args.T < 0.9:
            val_cnt += 1
            p2v_basket_file_validation.write(','.join(map(str, 
              [val_cnt, i, j, 1, 1, 0, t, basket_id])) + '\n')
          else:
            test_cnt += 1
            p2v_basket_file_test.write(','.join(map(str, 
              [test_cnt, i, j, 1, 1, 0, t, basket_id])) + '\n')
          

#      baskets_file.write(','.join(map(str, [basket_id, i, t] + basket)) + '\n')
      baskets_file.write(','.join(map(str, basket)) + '\n')
      if float(t)/args.T < 0.8:
        baskets_file_train.write(','.join(map(str, basket)) + '\n')
      elif float(t)/args.T < 0.9:
        baskets_file_validation.write(','.join(map(str, basket)) + '\n')
      else:
        baskets_file_test.write(','.join(map(str, basket)) + '\n')
      
#      baskets_i.append(basket)
#    baskets_it.append(baskets_i)

  print(timer() - start)
  baskets_file.close()
  baskets_file_train.close()
  baskets_file_test.close()
  baskets_file_validation.close()
  p2v_basket_file.close()
  p2v_basket_file_train.close()
  p2v_basket_file_test.close()
  p2v_basket_file_validation.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-I", type = int, help = "Number of consumers", 
                      default = 100)
  parser.add_argument("-T", type = int, help = "Number of weeks", 
                      default = 50)
  parser.add_argument("-C", type = int, help = "Number of categories",
                      default = 20)
  parser.add_argument("-Jc", type = int, 
                      help = "Number of products per categories",
                      default = 15)
  parser.add_argument("-Gamma0", type = float, 
                      help = "Category purchase incidence base utility",
                      default = -0.5)
  parser.add_argument("-Sigma0", type = float, 
                      help = "Standard deviation for error term in MNP",
                      default = 1)
  parser.add_argument("-tau0", type = float, 
                      help = "Standard deviation for covariance matrices in MNP",
                      default = 2)
  parser.add_argument("-Beta", type = float, 
                      help = "Price sensitivity",
                      default = 2)
  parser.add_argument("-Nneg", type = int, 
                      help = "# of negative samples",
                      default = 20)
  parser.add_argument("-iter", type = int, 
                      help = "# of iterations",
                      default = 10)
  parser.add_argument("-pow", type = float, 
                      help = "power",
                      default = 0.75)
  parser.add_argument("-output", type = str, 
                      help = "file to store output",
                      default = 'baskets')
  parser.add_argument("-data-split", type = str, 
                      help = "data split into train-test-validation fraction",
                      default = '0.8,0.1,0.1')
  
  args = parser.parse_args()
  data_generator(args)



if __name__ == "__main__":
  main()