import basket_completion as bc
import basket_generation as bg
import basket_reconstruction as br
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas
import warnings
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, "../node2vec_embeddings_modified/src")
import node2vec_main as n2v

sys.path.insert(0, "../p2v-map")
import p2v_main as p2v

from timeit import default_timer as timer
from multiprocessing import Pool as mpPool

np.set_printoptions(precision = 4, 
                    suppress = True, 
                    linewidth=500, 
                    #threshold=sys.maxsize,
                    edgeitems=8
                    )

selection = 'average'  #average / max / top_2
method='node2v' #'node2v' / 'p2v' / 'random' 
dimensions = 128
generate_baskets = False
build_graph = generate_baskets or False
overwrite_embedding = False
parallel = True
filename = 'out_files/embedding_%s_%s_p%s_q%s_minlen%s.npy' % ('%s', method, '%d', '%g', '%d')

if generate_baskets:
  print('basket generation')
  bg.main(I=200,
          T=100,
          C=20,
          Jc=15,
          Gamma0=0.5)

def embed_and_basket_completion(p, q, n):

    if generate_baskets:
      N = 20*15
    N = 300#1929 / 300
    min_basket_len = n
    #sim_v2 1500
    #sim 300

    if build_graph and method == 'node2v':
      embedding = n2v.main(input='data_graph/baskets_train.csv',
                          input_format= 'basketlist',
                          dimensions=3,
                          walk_length=3,
                          output='../node2vec_embeddings_modified/emb/baskets_train.emd',
                          overwrite=True,
                          overwrite_walks=True,
                          overwrite_transitions=True,
                          num_walks=2,
                          window_size=10,
                          iter=1,
                          p=p, 
                          q=q,
                          N=N,
                          min_basket_len=min_basket_len,
                          num_basket=100000)
    if method == 'node2v':
      if not overwrite_embedding:
        embedding = np.load(filename%('v', p, q, n))
      else:
        embedding = n2v.main(input='data_graph/baskets.graph',
                            input_format= 'basketgraph',
                            dimensions=dimensions,
                            walk_length=50,
                            output='../node2vec_embeddings_modified/emb/baskets_train.emd',
                            overwrite=True,
                            overwrite_walks=True,
                            overwrite_transitions=True,
                            num_walks=100,
                            window_size=n,
                            iter=5,
                            p=p, 
                            q=q,
                            N=N,
                            min_basket_len=min_basket_len)
        print('embedding shape', embedding.shape)
        np.save(filename%('v', p, q, n), embedding)
        
    ###for p2v comment n2v and uncomment this
    elif method == 'p2v':
      if not overwrite_embedding:
        embedding = (np.load(filename%('v', p, q, n)), 
                     np.load(filename%('w', p, q, n)))
      else:
        embedding = p2v.main(data_dir = 'p2v',
                             output_dir = '../p2v-map/results',
                             control_dir = '../p2v-map/control',
                             dimensions=dimensions,
                             p=p, 
                             q=q,
                             N=N,
                             min_basket_len=min_basket_len)
        np.save(filename%('v', p, q, n), embedding[0])
        np.save(filename%('w', p, q, n), embedding[1])
    ###for random
    else:
      embedding = np.array([]), np.array([])
    #basket completion
    '''
    acc = bc.basket_completion_accuracy(embedding=embedding,
                                        evaluation_file='train', 
                                        plot_embedding=False,
                                        clustering=False,
                                        n_clusters=20,
                                        p=p, 
                                        q=q,
                                        N=N,
                                        selection=selection,
                                        method=method, 
                                        n_sample=300-min_basket_len,
                                        bootstrap=10,
                                        min_basket_len=min_basket_len)
    '''
    #basket reconstruction
    acc = br.basket_completion_accuracy(input='data_graph/baskets.graph',
                                        embedding=embedding,
                                        evaluation_file='test', 
                                        plot_embedding=False,
                                        clustering=False,
                                        n_clusters=20,
                                        p=p, 
                                        q=q,
                                        N=N,
                                        selection=selection,
                                        method=method, 
                                        n_sample=300-min_basket_len,
                                        bootstrap=10,
                                        min_basket_len=min_basket_len)
    return acc

def plot_accuracy(p_q, acc_l):
  df = pd.DataFrame([(*i,*j) for (i,j) in zip(p_q,acc_l)], 
                    columns=['p', 'q', 'n', 'accuracy', 'lower', 'median', 'upper']
                    ).set_index(['n','q'])
  df['CI'] = df['upper'] - df['lower']
  df.to_csv(os.path.join('out_files', 'accuracy_'+method+'_'+selection+'.csv'))
  print(df)
  plt.clf()
  f = df['accuracy'].unstack(level=1).plot.bar(
          yerr=np.stack([(df['accuracy']-df['lower']).unstack(level=1).to_numpy().T,
                         (df['upper']-df['accuracy']).unstack(level=1).to_numpy().T]).transpose(1,0,2),
          capsize=4).get_figure()
  f.savefig(os.path.join('images', 'accuracy_'+method+'_'+selection+'.pdf'))


result_list = []
def log_result(result):
        result_list.append(result)

def main():

  p_range = [1000000]
  q_range = [0.2, 0.5 ,1 , 2, 4]
  q_range = [1.2, 1.4, 1.6, 1.8]
  q_range = [0.2, 0.5 ,1 ,1.2, 1.4, 1.6, 1.8, 2, 4]
  n_range = [2,3,4,5]
  q_range = [0.5]
  q_range = [0.2, 0.5, 1 ,1.2, 1.4, 1.6, 1.8, 2, 4]
  n_range = [2,3,4,5,6,7]
  q_range = [1,2]
  n_range = np.arange(2,8)
  p_q = list(itertools.product(p_range, q_range, n_range))
  print(p_q)
  acc_l = []
    
  if parallel:
    nCPU = min([6, len(p_q)])
    print('nCPU: {}'.format(nCPU))
    pool = mpPool(nCPU)
    acc = pool.starmap_async(embed_and_basket_completion, p_q)
    acc_l = acc.get()
    pool.close()
    pool.join()
  else:
    for (p,q,n) in p_q:
      start = timer()
      print('p = {}, q = {}, n = {}'.format(p,q,n))
      acc = embed_and_basket_completion(p, q, n)
      acc_l.append(acc)
      print('loop time {}'.format(timer()-start))

  plot_accuracy(p_q, acc_l)
  
if __name__ == "__main__":
  main()