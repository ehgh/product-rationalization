import os
import argparse
import numpy as np
from timeit import default_timer as timer
from os.path import join as join_path
from numpy.random import choice
from random import randrange
from random import choice as randchoice

np.set_printoptions(precision = 1, suppress = True)


#calculate product penetration weights for sampling purposes
def product_penetration_calculator(args, data_directory):
  product_penetration_weights = [0] * args.C * args.Jc
  with open(join_path(data_directory, args.output + '_train.csv'), 
    'r') as baskets_test:
    basket_cnt = len(baskets_test.readlines()) - 1 #reduce header count
  with open(join_path(data_directory, args.output + '_train.csv'), 
    'r') as baskets_test:
    next(baskets_test)
    for line in baskets_test:
      if line.strip():
        #use this line for def 1
        #basket = list(map(int, line.strip().split(',')))
        #use this line for def 2
        basket = list(set(map(int, line.strip().split(','))))
        for item in basket:
          product_penetration_weights[item] += 1 
  #def 1: penetration = number of item sold/number of all items sold
  #product_penetration_weights_probs = [i/sum(
  #  product_penetration_weights) for i in product_penetration_weights]
  
  #def 2: penetration = normalize(number of baskets containing the item/number of baskets)
  product_penetration_weights_probs = [i/basket_cnt 
    for i in product_penetration_weights]
  #normalize the probability vector
  product_penetration_weights_probs = [i / sum(product_penetration_weights_probs) for
    i in product_penetration_weights_probs]
  return product_penetration_weights_probs


#predict the missing item from a list of draws using remaining basket items
def predict_item(basket, draw):
  return randchoice(draw)


#create sampled data to predict missing item
def draw_samples(args, data_directory, output_directory):
  
  #create data directory and delete content of files if already exist
  if not os.path.isdir(output_directory):
    os.mkdir(output_directory)
  open(join_path(output_directory, args.output + '_baskets2predict.csv'), 'w').close()
  open(join_path(output_directory, args.output + '_draws.csv'), 'w').close()
  open(join_path(output_directory, args.output + '_items2predict.csv'), 'w').close()
  #open files to save data  
  draws = open(join_path(output_directory, args.output + '_draws.csv'), 'a')
  items2predict = open(join_path(output_directory, args.output + '_items2predict.csv'), 'a')
  baskets2predict = open(join_path(output_directory, args.output + '_baskets2predict.csv'), 'a')

  number_of_items_to_pick = args.n_sample

  #calculate product penetration weights for sampling purposes
  product_penetration_weights_probs = product_penetration_calculator(args, 
                                      data_directory)

  correct_predictions_cnt = 0
  instance_cnt = 0
  #generate samples
  with open(join_path(data_directory, args.output + '_test.csv'), 
    'r') as baskets_test:
    next(baskets_test)
    for line in baskets_test:
      if line.strip():
        basket = list(map(int, line.strip().split(',')))
        #remove baskets of small sizes
        if len(basket) > args.min_basket_len:
          #randomly select a product from basket to remove and predict
          idx2remove = randrange(len(basket))
          item2predict = basket.pop(idx2remove)
          #sample a pool of products to predict the removed item from them
          #sample from all products
          draw = list(choice(range(len(product_penetration_weights_probs)), 
            number_of_items_to_pick,
            p = product_penetration_weights_probs))
          draw.append(item2predict)
          #sample from the category of removed item
          #TO DO
          #........................................

          #predict the removed item from the sampled pool of products
          predict = predict_item(basket, draw)
          instance_cnt += 1
          if predict == item2predict:
            correct_predictions_cnt += 1

          #save the data
          draws.write(','.join(map(str, draw)) + '\n')
          items2predict.write(str(item2predict) + '\n')
          baskets2predict.write(','.join(map(str, basket)) + '\n')

  print(instance_cnt)
  print(1.0/(number_of_items_to_pick + 1))
  print(correct_predictions_cnt/instance_cnt)
  draws.close()
  items2predict.close()
  baskets2predict.close()

  return draw

def data_generator(args):
  
  #directories
  output_directory = 'basket_completion_data'
  data_directory = 'data'

  start = timer()

  #create sampled data to predict missing item
  draw = draw_samples(args, data_directory, output_directory)








  print(timer() - start)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-n-sample", type = int, 
                      help = "number of items to sample", 
                      default = 14)
  parser.add_argument("-min-basket-len", type = int, 
                      help = "minimum length of baskets", 
                      default = 5)
  

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