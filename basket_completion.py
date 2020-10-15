import sys
import os
import argparse
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from os.path import join as join_path
from random import randrange
from random import choice as randchoice


np.set_printoptions(precision = 1, suppress = True)


#calculate product penetration weights for sampling purposes
def product_penetration_calculator(args, data_directory):
  product_penetration_weights = [0] * args.C * args.Jc
  basket_cnt = 0
  with open(join_path(data_directory, args.output + '_train.csv'), 
    'r') as baskets_test:
    next(baskets_test)
    for line in baskets_test:
      if line.strip():
        #use this line for def 1
        #basket = list(map(int, line.strip().split(',')))
        #use this line for def 2
        basket = list(set(map(int, line.strip().split(','))))
        basket_cnt += 1
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

def product_penetration_calculator_v2(args, data_directory):
  product_penetration_weights = np.zeros(args.C * args.Jc)
  basket_cnt = 0
  with open(join_path(data_directory, args.output + '_train.csv'), 
    'r') as baskets_test:
    next(baskets_test)
    for line in baskets_test:
      if line.strip():
        #use this line for def 1
        #basket = list(map(int, line.strip().split(',')))
        #use this line for def 2
        basket = list(set(map(int, line.strip().split(','))))
        basket_cnt += 1
        product_penetration_weights[basket] += 1 
  #def 1: penetration = number of item sold/number of all items sold
  #product_penetration_weights_probs = [i/sum(
  #  product_penetration_weights) for i in product_penetration_weights]
  
  #def 2: penetration = normalize(number of baskets containing the item/number of baskets)
  product_penetration_weights_probs = product_penetration_weights/basket_cnt
  #normalize the probability vector
  product_penetration_weights_probs = product_penetration_weights_probs/product_penetration_weights_probs.sum()
  return product_penetration_weights_probs


#predict the missing item from a list of draws using remaining basket items
def predict_item(args, basket, draw, method = 'random', select = 'max', **kwargs):
  
  if method == 'random':
    return randchoice(draw)
  elif method == 'p2v':
    (p2v_embeddings_v, p2v_embeddings_w) = kwargs['embedding']
  elif method == 'node2v':
    (node2v_embedding, denom) = kwargs['embedding']

  #first item of basket is the one to predict
  if method == 'p2v' or method == 'random':
    basket_embeds_v = p2v_embeddings_v[basket[1:], :]
    if select == 'average':
      predicted_item = draw[np.dot(basket_embeds_v, p2v_embeddings_w[draw].T).mean(0).argmax()]
    else:
      predicted_item = draw[np.dot(basket_embeds_v, p2v_embeddings_w[draw].T).max(0).argmax()]
  elif method =='node2v':
    basket_embeds_v = node2v_embedding[basket[1:], :]
    denom_v = denom[basket[1:]][:,None]
    if select == 'average':
      predicted_item = draw[(np.exp(np.dot(basket_embeds_v, node2v_embedding[draw].T))/denom_v).mean(0).argmax()]
    else:
      predicted_item = draw[(np.exp(np.dot(basket_embeds_v, node2v_embedding[draw].T))/denom_v).max(0).argmax()]
  
  return predicted_item


#create sampled data to predict missing item
def draw_samples(args, data_directory, output_directory):
  
  #create data directory and delete content of files if already exist
  if not os.path.isdir(output_directory):
    os.mkdir(output_directory)
  open(join_path(output_directory, args.output + '_baskets2predict.csv'), 'w').close()
  open(join_path(output_directory, args.output + '_draws.csv'), 'w').close()
  #open files to save data  
  draws = open(join_path(output_directory, args.output + '_draws.csv'), 'a')
  baskets2predict = open(join_path(output_directory, args.output + '_baskets2predict.csv'), 'a')

  number_of_items_to_pick = args.n_sample

  #calculate product penetration weights for sampling purposes
  product_penetration_weights_probs = product_penetration_calculator(args, 
                                      data_directory)
  all_items = np.arange(len(product_penetration_weights_probs))
  
  #load p2v embeddings
  #p2v_embeddings_v = pd.read_csv(args.p2v_embedding_v, sep = ',')[['x', 'y']].to_numpy()
  #p2v_embeddings_w = pd.read_csv(args.p2v_embedding_w, sep = ',')[['x', 'y']].to_numpy()
  p2v_embeddings_v = np.load(args.p2v_v)
  p2v_embeddings_w = np.load(args.p2v_w)
  print(p2v_embeddings_v.shape)
  
  item2predict, predictions = [], []
  #generate samples
  start = timer()
  with open(join_path(data_directory, args.output + '_test.csv'), 
    'r') as baskets_test_f:
    next(baskets_test_f)
    baskets_test = [list(map(int, x.strip().split(','))) for x in baskets_test_f.readlines() if x.strip()]
  
  for instance_cnt, basket in enumerate(baskets_test):
      
      #remove baskets of small sizes
      if len(basket) > args.min_basket_len:

          #shuffle the basket to predict the first item of it
          random.shuffle(basket)
          #sample a pool of products to predict the removed item from them
          #sample from all products
          draw = np.random.choice(all_items, 
            number_of_items_to_pick, replace=False,
            p = product_penetration_weights_probs).tolist()
          draw.append(basket[0])
          #sample from the category of removed item
          #TO DO
          #........................................

          #predict the removed item from the sampled pool of products
          method = 'node2v' #'node2v' or 'p2v' or 'random'
          if method == 'p2v' or method == 'random':
            predict = predict_item(args, basket, draw, method, select = 'average', embedding = (p2v_embeddings_v, p2v_embeddings_w))
          elif method =='node2v':
            embedding = p2v_embeddings_v
            denom = np.exp(embedding @ embedding.T).sum(0)
            predict = predict_item(args, basket, draw, method, select = 'average', embedding = (embedding, denom))

          if instance_cnt % 1000 == 0:
            print(instance_cnt) 
            print(timer() - start)
            start = timer()

          item2predict.append(basket[0])
          predictions.append(predict)
          
          #save the data
          #draws.write(','.join(map(str, draw)) + '\n')
          #baskets2predict.write(','.join(map(str, basket)) + '\n')

  print(1.0/(number_of_items_to_pick + 1))
  print(sum(x==y for x, y in zip(item2predict, predictions))/len(predictions))
  draws.close()
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
  parser.add_argument("-p2v-embedding-v", type = str, 
                      help = "path to p2v-v embedding file", 
                      default = 'data/embeddings_wi.csv')
  parser.add_argument("-p2v-embedding-w", type = str, 
                      help = "path to p2v-w embedding file", 
                      default = 'data/embeddings_wo.csv')
  parser.add_argument("-p2v-v", type = str, 
                      help = "path to p2v v embeddings file", 
                      default = 'data/wi.npy')
  parser.add_argument("-p2v-w", type = str, 
                      help = "path to p2v w embeddings file", 
                      default = 'data/wo.npy')
  

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