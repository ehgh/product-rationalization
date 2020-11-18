import basket_completion as bc
import sys
sys.path.insert(0, "../node2vec_embeddings_modified/src")
import node2vec_main as n2v
import itertools
from timeit import default_timer as timer

def main():
  p_range = [100, 1000000]
  q_range = [1, 2, 4, 8, 16, 100, 10000]
  p_q = list(itertools.product(p_range, q_range))
  print(p_q)
  acc = []
  for (p,q) in p_q:
    start = timer()
    print('p = {}, q = {}'.format(p,q))
    embedding = n2v.main(input='../node2vec_embeddings_modified/graph/baskets.graph',
                        input_format= 'basketgraph',
                        dimensions=6,
                        walk_length=20,
                        output='../node2vec_embeddings_modified/emb/baskets_train.emd',
                        overwrite=False,
                        p=p, 
                        q=q)
    acc.append(bc.basket_completion_accuracy(embedding=embedding))
    print('loop time {}'.format(timer()-start))

  print([i for i in zip(p_q,acc)])


if __name__ == "__main__":
  main()