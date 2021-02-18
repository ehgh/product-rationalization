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


np.set_printoptions(precision = 0, 
                    suppress = True, 
                    linewidth=500, 
                    #threshold=sys.maxsize,
                    edgeitems=8
                    )


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
    denom_v = denom[draw]#denom[basket[1:]]
    if select == 'average':
      predicted_item = draw[(np.dot(basket_embeds_v, node2v_embedding[draw].T).sum(0)-np.log(denom_v)*len(basket)).argmax()]
      #predicted_item = draw[(np.dot(basket_embeds_v, node2v_embedding[draw].T).sum(0)-np.log(denom_v)).argmax()]
      #predicted_item = draw[(np.dot(basket_embeds_v, node2v_embedding[draw].T).sum(0)).argmax()]
      #predicted_item = draw[((np.abs(basket_embeds_v[:,None,:] - node2v_embedding[draw])).sum(-1).sum(0)).argmax()]
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
  product_penetration_weights_probs = np.array(product_penetration_weights_probs)
  all_items = np.arange(args.C * args.Jc)
  
  #load p2v embeddings
  if 'embedding' in vars(args) and args.embedding.any():
    p2v_embeddings_v = args.embedding
    p2v_embeddings_w = args.embedding
  else:
    #p2v_embeddings_v = pd.read_csv(os.path.join(data_directory, args.p2v_embedding_v), sep = ',')[['x', 'y']].to_numpy()
    #p2v_embeddings_w = pd.read_csv(os.path.join(data_directory, args.p2v_embedding_w), sep = ',')[['x', 'y']].to_numpy()
    p2v_embeddings_v = np.load(os.path.join(data_directory, args.p2v_v))
    p2v_embeddings_w = np.load(os.path.join(data_directory, args.p2v_w))
  print('basket completion {}'.format(p2v_embeddings_v.shape))
  
  #plot embeddings and score heatmaps
  if args.plot_embedding == 'True':
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    im_00 = ax[0,0].imshow(p2v_embeddings_v.T, cmap='hot', interpolation='nearest', aspect=10)
    im_01 = ax[0,1].imshow(p2v_embeddings_w.T, cmap='hot', interpolation='nearest', aspect=10)
    fig.colorbar(im_00, ax=ax[0,0])  
    fig.colorbar(im_01, ax=ax[0,1])  
    p2v_embeddings_v_normalized = p2v_embeddings_v / np.linalg.norm(p2v_embeddings_v, axis=1)[:,None]
    p2v_embeddings_w_normalized = p2v_embeddings_w / np.linalg.norm(p2v_embeddings_w, axis=1)[:,None]
    im_10 = ax[1,0].imshow(p2v_embeddings_v_normalized.T, cmap='hot', interpolation='nearest', aspect=10)
    im_11 = ax[1,1].imshow(p2v_embeddings_w_normalized.T, cmap='hot', interpolation='nearest', aspect=10)
    fig.colorbar(im_10, ax=ax[1,0])  
    fig.colorbar(im_11, ax=ax[1,1])  
    plt.show()
    p2v_v_w = np.dot(p2v_embeddings_v,p2v_embeddings_w.T) + 3.09
    plt.imshow(p2v_v_w, cmap='hot', interpolation='nearest', aspect='equal')
    plt.colorbar()
    plt.show()

  #for node2vec method
  denom = np.exp(p2v_embeddings_v @ p2v_embeddings_v.T).sum(0)

  item2predict, predictions = [], []
  #generate samples
  start = timer()
  with open(join_path(data_directory, args.output + '_test.csv'), 
    'r') as baskets_test_f:
    next(baskets_test_f)
    baskets_test = [list(map(int, x.strip().split(','))) for x in baskets_test_f.readlines() if x.strip()]

  #bakset sizes
  basket_len = []
  for instance_cnt, basket in enumerate(baskets_test):
      
      #remove duplicates from the basket if necessary
      #basket = list(set(basket))
      
      basket_len.append(len(basket))

      #remove baskets of small sizes
      if len(basket) > args.min_basket_len:
          #shuffle the basket to predict the first item of it
          random.shuffle(basket)
          #sample a pool of products to predict the removed item from them
          #sample from all products minus the ones already in the basket
          remaining_items = np.setdiff1d(all_items, basket, assume_unique=True)
          remaining_weights = product_penetration_weights_probs[remaining_items]
          draw = np.random.choice(remaining_items, 
            number_of_items_to_pick, replace=False,
            p = remaining_weights/remaining_weights.sum()).tolist()
          draw.append(basket[0])
          #sample from the category of removed item
          #TO DO
          #........................................

          #predict the removed item from the sampled pool of products
          method = 'p2v' #'node2v' or 'p2v' or 'random'
          #selection criterion: 'average' or 'max'
          selection = 'max'
          if method == 'p2v' or method == 'random':
            predict = predict_item(args, basket, draw, method, select = selection, embedding = (p2v_embeddings_v, p2v_embeddings_w))
          elif method =='node2v':
            predict = predict_item(args, basket, draw, method, select = selection, embedding = (p2v_embeddings_v, denom))

          if instance_cnt % 100000 == 0:
            print(instance_cnt) 
            print(timer() - start)
            start = timer()

          item2predict.append(basket[0])
          predictions.append(predict)
          
          #save the data
          #draws.write(','.join(map(str, draw)) + '\n')
          #baskets2predict.write(','.join(map(str, basket)) + '\n')

  ##show the CDF of basket sizes
  #import matplotlib.pyplot as plt
  #basket_len = np.array(basket_len)
  #n_bins = 40
  #_ = plt.hist(basket_len, n_bins, density=True, histtype='step',
  #                         cumulative=True, label='Empirical')
  #plt.show()


  print('random probability', 1.0/(number_of_items_to_pick + 1))
  prediciton_accuracy = sum(x==y for x, y in zip(item2predict, predictions))/len(predictions)
  print('prediciton accuracy', prediciton_accuracy)
  draws.close()
  baskets2predict.close()

  return prediciton_accuracy

def data_generator(args):
  
  #directories
  output_directory = 'basket_completion_data'
  data_directory = 'data_graph'

  start = timer()

  #create sampled data to predict missing item
  prediciton_accuracy = draw_samples(args, data_directory, output_directory)

  return prediciton_accuracy







  print(timer() - start)


def basket_completion_accuracy(**kwargs):
  parser = argparse.ArgumentParser()
  parser.add_argument("-n-sample", type = int, 
                      help = "number of items to sample", 
                      default = 14)
  parser.add_argument("-min-basket-len", type = int, 
                      help = "minimum length of baskets", 
                      default = 5)
  parser.add_argument("-p2v-embedding-v", type = str, 
                      help = "path to p2v-v embedding file", 
                      default = 'embeddings_wi.csv')
  parser.add_argument("-p2v-embedding-w", type = str, 
                      help = "path to p2v-w embedding file", 
                      default = 'embeddings_wo.csv')
  parser.add_argument("-p2v-v", type = str, 
                      help = "path to p2v v embeddings file", 
                      default = 'wi.npy')
  parser.add_argument("-p2v-w", type = str, 
                      help = "path to p2v w embeddings file", 
                      default = 'wo.npy')
  parser.add_argument("-plot-embedding", type = str, 
                      help = "flag to plot embedding heatmaps", 
                      default = 'False')
  

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
  args_dict = vars(args)
  for k, v in kwargs.items():
    args_dict[k] = v

  return data_generator(args)



if __name__ == "__main__":
  basket_completion_accuracy()