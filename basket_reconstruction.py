import sys
import os
import argparse
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle
import networkx as nx

from timeit import default_timer as timer
from os.path import join as join_path
from random import randrange
from random import choice as randchoice
from tqdm import tqdm
from sklearn.cluster import SpectralClustering, DBSCAN
from random import choices

sys.path.insert(0, "../node2vec_embeddings_modified/src")
import node2vec

np.set_printoptions(precision = 5, 
                    suppress = True, 
                    linewidth=500, 
                    #threshold=sys.maxsize,
                    edgeitems=8
                    )


#calculate product penetration weights for sampling purposes
def product_penetration_calculator(args, data_directory):
  product_penetration_weights = [0] * args.N
  basket_cnt = 0
  with open(join_path(data_directory, args.output + '_train.csv'), 
    'r') as baskets_test:
    next(baskets_test)
    for line in baskets_test:
      if line.strip():
        #use this line for def 1 
        #(if products do not repeat per basket the two def become the same)
        basket = list(map(int, line.strip().split(',')))
        #use this line for def 2
        #basket = list(set(map(int, line.strip().split(','))))
        basket_cnt += 1
        for item in basket:
            try:
              product_penetration_weights[item] += 1 
            except:
              print('exception happened, item: {}'.format(item))
  #def 1: penetration = number of item sold/number of all items sold
  #product_penetration_weights_probs = [i/sum(
  #  product_penetration_weights) for i in product_penetration_weights]
  
  #def 2: penetration = normalize(number of baskets containing the item/number of baskets)
  #this line is unnecessary since next line normalizes it anyways
  #product_penetration_weights_probs = [i/basket_cnt 
  #  for i in product_penetration_weights]
  #normalize the probability vector
  product_penetration_weights_probs = [i / sum(product_penetration_weights) for
    i in product_penetration_weights]
  
  return product_penetration_weights_probs

def product_penetration_calculator_v2(args, data_directory):
  product_penetration_weights = np.zeros(args.N)
  basket_cnt = 0
  with open(join_path(data_directory, args.output + '_train.csv'), 
    'r') as baskets_test:
    next(baskets_test)
    for line in baskets_test:
      if line.strip():
        #use this line for def 1
        basket = list(map(int, line.strip().split(',')))
        #use this line for def 2
        #basket = list(set(map(int, line.strip().split(','))))
        basket_cnt += 1
        product_penetration_weights[basket] += 1 
  #def 1: penetration = number of item sold/number of all items sold
  #product_penetration_weights_probs = [i/sum(
  #  product_penetration_weights) for i in product_penetration_weights]
  
  #def 2: penetration = normalize(number of baskets containing the item/number of baskets)
  #this line is unnecessary since next line normalizes it anyways
  #product_penetration_weights_probs = product_penetration_weights/basket_cnt
  #normalize the probability vector
  product_penetration_weights_probs = product_penetration_weights/product_penetration_weights.sum()
  return product_penetration_weights_probs

def load_embedding(args):

  #placeholder for model
  model = 0
  if args.method == 'node2v':
    embeddings_v = args.embedding
    #embeddings_w = args.embedding
    model = pickle.load(open('out_files/model_p'+str(args.p)+'_q'+str(args.q)+'_minlen'+str(args.min_basket_len)+'.pkl', 'rb'))
    #get wi vectors
    embeddings_w = model.trainables.syn1neg
    #get the item's indices
    index2entity = np.array([int(i) for i in model.wv.index2entity if i])
    #sort wi from 0-N
    embeddings_w = embeddings_w[index2entity.argsort()]
    #find missing products in vocab
    missing_prods = np.flatnonzero(~np.in1d(np.arange(args.N),index2entity))
    missing_prods -= np.arange(missing_prods.size)
    #insert all zeros feature vector for missing products
    embeddings_w = np.insert(embeddings_w, missing_prods, np.zeros(embeddings_w.shape[1]), 0)    
  elif args.method == 'p2v':
    embeddings_v, embeddings_w = args.embedding
  elif args.method == 'random':
    embeddings_v, embeddings_w = args.embedding
  print('embeddings_v shape: {}'.format(embeddings_v.shape))
  
  return embeddings_v, embeddings_w, model
  
def load_graph(args):
  nx_G = nx.readwrite.gpickle.read_gpickle(args.input+'_minlen'+str(args.min_basket_len))
  G = node2vec.Graph(nx_G, False, args.p, args.q)
  with open('out_files/alias_nodes_p'+str(args.p)+'_q'+str(args.q)+'_minlen'+str(args.min_basket_len)+'.pkl', 'rb') as f:
      G.alias_nodes = pickle.load(f)
  with open('out_files/alias_edges_p'+str(args.p)+'_q'+str(args.q)+'_minlen'+str(args.min_basket_len)+'.pkl', 'rb') as f:
      G.alias_edges = pickle.load(f)

  return G

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
def plot_clusters(embedding, sc):
  print('sccccccccccccccc',sc)
  print(sc.labels_)
  if hasattr(sc, 'labels_'):
      y_pred = sc.labels_.astype(int)
  else:
      y_pred = sc.predict(X)

  plt.subplot(len(embedding), 1, 1)
  plt.title('Clusters', size=18)

  colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                       '#f781bf', '#a65628', '#984ea3',
                                       '#999999', '#e41a1c', '#dede00']),
                                int(max(y_pred) + 1))))
  # add black color for outliers (if any)
  colors = np.append(colors, ["#000000"])
  plt.scatter(embedding[:, 0], embedding[:, 1], s=10, color=colors[y_pred])

  plt.xlim(-2.5, 2.5)
  plt.ylim(-2.5, 2.5)
  plt.xticks(())
  plt.yticks(())
  plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
           transform=plt.gca().transAxes, size=15,
           horizontalalignment='right')


#predict the missing item from a list of draws using remaining basket items
def predict_item(args, basket, draw, **kwargs):
  
  random.shuffle(draw)
  
  method = args.method
  if method == 'random':
    rand_choice = randchoice(draw)
    return rand_choice, rand_choice
  else:
    (embeddings_v, embeddings_w) = kwargs['embedding']
    model = kwargs['model']

  #first item of basket is the one to predict
  if method == 'p2v' or method =='node2v':
    basket_embeds_v = embeddings_v[basket[1:], :]
    basket_embeds_w = embeddings_w[basket[1:], :]
    if args.selection == 'average':
      predicted_item1 = draw[np.dot(basket_embeds_v, embeddings_w[draw].T).mean(0).argmax()]
      predicted_item2 = draw[np.dot(embeddings_v[draw], basket_embeds_w.T).mean(1).argmax()]
    elif args.selection == 'max':
      predicted_item1 = draw[np.dot(basket_embeds_v, embeddings_w[draw].T).max(0).argmax()]
      predicted_item2 = draw[np.dot(embeddings_v[draw], basket_embeds_w.T).max(1).argmax()]
    elif args.selection.startswith('top'):
      m = min(int(args.selection[4:]), len(basket)-1)
      predicted_item1 = draw[np.partition(np.dot(basket_embeds_v, embeddings_w[draw].T), m-1, axis=0)[:-m-1:-1,:].sum(0).argmax()]
      predicted_item2 = draw[np.partition(np.dot(embeddings_v[draw], basket_embeds_w.T), m-1, axis=1)[:,:-m-1:-1].sum(1).argmax()]

  return predicted_item1, predicted_item2
  

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

  #calculate product penetration weights for sampling purposes
  product_penetration_weights_probs = product_penetration_calculator(args, 
                                      data_directory)
  product_penetration_weights_probs = np.array(product_penetration_weights_probs)
  all_items = np.arange(args.N)
  
  #load embeddings
  embeddings_v, embeddings_w, model = load_embedding(args)

  #for node2vec method
  #denom = np.exp(embeddings_v @ embeddings_w.T).sum(0)[:,None]
  #denom = (embeddings_v @ embeddings_w.T).sum(0)[:,None]

  #plot embeddings and score heatmaps
  if args.plot_embedding:
    plot_embedding = False
    if plot_embedding: 
      fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
      im_00 = ax[0,0].imshow(embeddings_v.T, cmap='seismic', interpolation='nearest', aspect=10)
      im_01 = ax[0,1].imshow(embeddings_w.T, cmap='seismic', interpolation='nearest', aspect=10)
      fig.colorbar(im_00, ax=ax[0,0])  
      fig.colorbar(im_01, ax=ax[0,1])
      embeddings_v_norm = np.linalg.norm(embeddings_v, axis=1)[:,None]  
      embeddings_v_norm[embeddings_v_norm==0] = 1
      embeddings_w_norm = np.linalg.norm(embeddings_w, axis=1)[:,None]  
      embeddings_w_norm[embeddings_w_norm==0] = 1
      embeddings_v_normalized = embeddings_v / embeddings_v_norm
      embeddings_w_normalized = embeddings_w / embeddings_w_norm
      im_10 = ax[1,0].imshow(embeddings_v_normalized.T, cmap='seismic', interpolation='nearest', aspect=10)
      im_11 = ax[1,1].imshow(embeddings_w_normalized.T, cmap='seismic', interpolation='nearest', aspect=10)
      fig.colorbar(im_10, ax=ax[1,0])  
      fig.colorbar(im_11, ax=ax[1,1])  
      #plt.show()
      fig.savefig(join_path('images', "embedding_"+args.method+"_p"+str(args.p)+"_q"+str(args.q)+"_minlen"+str(args.min_basket_len)+".pdf"), bbox_inches='tight')

    v_w_cross = np.dot(embeddings_v, embeddings_w.T)
    if args.clustering:
      v_w_cross -= v_w_cross.min()
      ###spectral clustering
      sc = SpectralClustering(n_clusters=args.n_clusters, 
                              affinity='precomputed', 
                              n_init=100, 
                              assign_labels='discretize'
                              )
      sc.fit_predict((v_w_cross+v_w_cross.T)/2.0)
      clstr = np.argsort(sc.labels_)
      #plot_clusters(embeddings_v, sc)
      ###DBSCAN clustering
      #v_w_cross = v_w_cross.max() - v_w_cross
      #sc = DBSCAN(eps=3, 
      #            min_samples=2, 
      #            metric='precomputed', 
      #            )
      #sc.fit_predict(v_w_cross)
      #clstr = np.argsort(sc.labels_)
      v_w_cross = v_w_cross[np.ix_(clstr, clstr)]
      
      f = plt.figure()
      plt.scatter(sc.labels_, np.arange(sc.labels_.size), cmap='seismic')  
      f.savefig(join_path('images', "clusters"+args.method+"_p"+str(args.p)+"_q"+str(args.q)+"_minlen"+str(args.min_basket_len)+"_normed.pdf"), bbox_inches='tight')
    
    ##standardize the v_w for histogram and shift center
    v_w_cross = (v_w_cross - v_w_cross.mean())/(v_w_cross.std())
    
    ##co-occurence score heatmap
    f = plt.figure()
    plt.imshow(v_w_cross[:300, :300], cmap='seismic', interpolation='nearest', aspect='equal')  
    plt.colorbar()
    f.savefig(join_path('images', "v_w_"+args.method+"_p"+str(args.p)+"_q"+str(args.q)+"_minlen"+str(args.min_basket_len)+"_normed.pdf"), bbox_inches='tight')
    
    ##histogram of coocurrence scores
    #f = plt.figure()
    #plt.hist(v_w_cross.flatten(), bins=np.arange(-25, 15, 0.2))
    #plt.yscale('log', nonposy='clip')
    #f.savefig(join_path('images', "v_w_"+args.method+"_"+args.selection+"_p"+str(args.p)+"_q"+str(args.q)+"_minlen"+str(args.min_basket_len)+"_normed_histogram.pdf"), bbox_inches='tight')
    
  #load training product graph
  #G = nx.readwrite.gpickle.read_gpickle(join_path(data_directory, 'baskets.graph'+'_minlen'+str(args.min_basket_len)))

  #generate samples
  start = timer()
  with open(join_path(data_directory, args.output + '_' + args.evaluation_file + '.csv'), 
    'r') as baskets_test_f:
    next(baskets_test_f)
    baskets_test_data = [list(map(int, x.strip().split(','))) for x in baskets_test_f.readlines() if x.strip()]
  print('-'*100)
  
  ##bootstrap 1000 times to get CI
  num_basket_test = len(baskets_test_data)
  #list to store bootsrap accuracy results
  prediciton_accuracy = []
  ##load graph for random walk reconstruction
  G = load_graph(args)
  nodes = G.G.nodes()

  for boot_i in tqdm(range(args.bootstrap)):
    item2predict, predictions1, predictions2 = [], [], []
    baskets_test = choices(baskets_test_data, k=max(num_basket_test, 250))

    #bakset sizes
    basket_len = []
    hitrate = []
    sales = np.zeros(args.N)
    sales_predict = np.zeros(args.N)
    cnt = 0
    for instance_cnt, basket in enumerate(baskets_test):
        
        #remove duplicates from the basket if necessary
        basket = list(set(basket))
        
        basket_len.append(len(basket))

        #remove baskets of small sizes
        if len(basket) == args.min_basket_len:
            cnt+=1
            
            #shuffle the basket to reconstruct basket from last item
            random.shuffle(basket)

            basket2fill = basket[-1:]
            draw = all_items.copy()
            while len(basket2fill) < len(basket):
              #sample a pool of products to predict the removed item from them
              #sample from all products minus the ones already in the basket
              draw = np.setdiff1d(draw, basket2fill[-1:], assume_unique=True)
              
              #predict the removed item from the sampled pool of products
              predict = predict_item(args, [-1]+basket2fill, draw, embedding = (embeddings_v, embeddings_w), model=model)
              #basket2fill.insert(0, predict[0])
              basket2fill.append(predict[0])
            '''
            ##random walk basket reconstruction
            if basket[-1] in nodes:
              basket2fill = G.simulate_walks(1, len(basket), nodes=basket[-1:])[0]
            else:
              continue
            '''
            hitrate.append((1-len(np.setdiff1d(basket2fill, basket, assume_unique=True))/(len(basket)-1))*100)
            #sales[basket] += 1
            #sales_predict[basket2fill] += 1
            

    ##show the CDF of basket sizes
    #basket_len = np.array(basket_len)
    #n_bins = 40
    #_ = plt.hist(basket_len, n_bins, density=True, histtype='step',
    #                         cumulative=False, label='Empirical')
    #plt.show()

    #idx = sales>1
    #idx = np.argpartition(sales, 20)[:-21:-1]
    #prediciton_accuracy.append((np.abs(sales[idx] - sales_predict[idx])/sales[idx]).mean())
    prediciton_accuracy.append(sum(hitrate)/len(hitrate))

  ##histogram of sales
  f = plt.figure()
  plt.hist(sales.flatten(), bins=np.arange(0, 300, 1))
  plt.xscale('log')
  f.savefig(join_path('images', "sales_"+args.method+"_"+args.selection+"_p"+str(args.p)+"_q"+str(args.q)+"_minlen"+str(args.min_basket_len)+"_normed_histogram.pdf"), bbox_inches='tight')

  print('number of baskets in test set with basket size {}: {}/{}'.format(args.min_basket_len, cnt, len(baskets_test_data)))

  print('random probability', 1.0/(args.n_sample + 1))
  prediciton_accuracy = np.array(prediciton_accuracy)
  accuracy = np.array([prediciton_accuracy.mean()])
  accuracy = np.hstack((accuracy, np.percentile(prediciton_accuracy, [5, 50, 95])))
  return accuracy

def data_generator(args):
  
  print('\n p = {}, q = {}, n = {}'.format(args.p,args.q,args.min_basket_len))

  #directories
  output_directory = 'basket_completion_data'
  data_directory = 'data_graph'

  #create sampled data to predict missing item
  prediciton_accuracy = draw_samples(args, data_directory, output_directory)

  return prediciton_accuracy


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
                      default = 'wi_full.npy')
  parser.add_argument("-p2v-w", type = str, 
                      help = "path to p2v w embeddings file", 
                      default = 'wo_full.npy')
  parser.add_argument("-plot-embedding", type = bool, 
                      help = "flag to plot embedding heatmaps", 
                      default = False)
  parser.add_argument("-method", type = str, 
                      help = "basket completion method (random/p2v/node2v) - default p2v", 
                      default = 'p2v')
  parser.add_argument("-selection", type = str, 
                      help = "selection method for basket completion (average/max) - default average", 
                      default = 'average')
  

  parser.add_argument("-N", type = int, help = "Number of products")
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
  if args.N is None:
    args_dict['N'] = args.C * args.Jc 

  return data_generator(args)



if __name__ == "__main__":
  basket_completion_accuracy()