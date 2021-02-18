import basket_completion as bc
import sys
import itertools
import numpy as np

sys.path.insert(0, "../node2vec_embeddings_modified/src")
import node2vec_main as n2v

from timeit import default_timer as timer
from multiprocessing import Pool as mpPool

np.set_printoptions(precision = 4, 
                    suppress = True, 
                    linewidth=500, 
                    #threshold=sys.maxsize,
                    edgeitems=8
                    )


def embed_and_basket_completion(p, q):

    N = 1929
    #sim_v2 1500
    #sim 300

    embedding = n2v.main(input='data_graph/baskets.graph',
                        input_format= 'basketgraph',
                        dimensions=128,
                        walk_length=50,
                        output='../node2vec_embeddings_modified/emb/baskets_train.emd',
                        overwrite=False,
                        overwrite_walks=False,
                        overwrite_transitions=False,
                        num_walks=20,
                        window_size=10,
                        iter=5,
                        p=p, 
                        q=q,
                        N=N)
    '''
    embedding = n2v.main(input='data_graph/baskets_train.csv',
                        input_format= 'basketlist',
                        dimensions=3,
                        walk_length=5,
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
                        min_basket_len=2,
                        num_basket=10000)
    '''
    ###for p2v comment n2v and uncomment this
    #embedding = np.array([])
    acc = bc.basket_completion_accuracy(embedding=embedding, 
                                        plot_embedding='True', 
                                        p=p, 
                                        q=q,
                                        N=N,
                                        selection='max',
                                        method='node2v', 
                                        n_sample=15,
                                        min_basket_len=2)

    return acc

result_list = []
def log_result(result):
        result_list.append(result)

def main():

  p_range = [1000000]
  q_range = [0.5,1,2]
  p_q = list(itertools.product(p_range, q_range))
  print(p_q)
  acc_l = []
  
  parallel = True
  
  if parallel:
    nCPU = min([5, len(p_q)])
    print(nCPU)
    pool = mpPool(nCPU)
    acc = pool.starmap_async(embed_and_basket_completion, p_q)
    acc_l = acc.get()
    pool.close()
    pool.join()
  else:
    for (p,q) in p_q:
      start = timer()
      print('p = {}, q = {}'.format(p,q))
      acc = embed_and_basket_completion(p, q)
      acc_l.append(acc)
      print('loop time {}'.format(timer()-start))

  print([i for i in zip(p_q,acc_l)])


if __name__ == "__main__":
  main()