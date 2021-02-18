import os
import argparse
import numpy as np
from scipy.linalg import block_diag
from timeit import default_timer as timer
from os.path import join as join_path
from scipy.stats import weibull_min

np.set_printoptions(precision = 2, suppress = True, linewidth=500)

#generate inter-category correlation matrix 
#purchasing categories is selected by MVN(Gamma0, Omega)
#output is in shape of (I * T * C):
#   I: number of consumers
#   T: number of weeks
#   C: number of products per category
def omega(): 
  D1 = [[1, -0.25, -0.25, -0.25],
        [-0.25, 1, -0.25, -0.25],
        [-0.25, -0.25, 1, -0.25],
        [-0.25, -0.25, -0.25, 1]] 
  D2 = [[1, -0.33, -0.28, -0.18],
        [-0.33, 1, -0.3, -0.16],
        [-0.28, -0.3, 1, -0.35],
        [-0.18, -0.16, -0.35, 1]]
  D3 = [[1, -0.32, -0.37, -0.46],
        [-0.32, 1, -0.35, 0.17],
        [-0.37, -0.35, 1, 0.45],
        [-0.46, 0.17, 0.45, 1]]
  D4 = [[1, 0.17, 0.39, 0.24],
        [0.17, 1, 0.36, 0.27],
        [0.39, 0.36, 1, 0.21],
        [0.24, 0.27, 0.21, 1]]
  D5 = [[1, 0.8],
        [0.8, 1]]
  D6 = [[1, 0.8],
        [0.8, 1]]
  mat = block_diag(D1, D2, D3, D4, D5, D6)
  return block_diag(mat, mat, mat, mat, mat)


#generates intra-category correlation matrix
#Sigma_c = (tau * I_c) * omega_c * (tau * I_c)
#omega_c ~ Beta(0.2, 1) (symmetric with diagonal 1)
def sigma_per_cat(args):
  Sigma_c = np.empty((args.C, args.Jc, args.Jc))
  sz = args.Jc
  tau = args.tau0
  I_c = np.identity(args.Jc)
  for category in range(args.C):
    b = np.random.beta(0.2, 1, size = (sz, sz))
    np.fill_diagonal(b, 1)
    #TODO: instead of dot in line below just fill in the top triangle
    #TODO: make sure value are not negative
    omega_c = np.dot(b, b.T)
    tau_I_c = tau * I_c
    sigma_c = np.matmul(tau_I_c, omega_c)
    sigma_c = np.matmul(sigma_c, tau_I_c)
    Sigma_c[category, :, :] = sigma_c
  return Sigma_c


def sigma_per_cat_vine(args, eta=0):
  Sigma_c = np.empty((args.C, args.Jc, args.Jc))
  sz = args.Jc
  tau = args.tau0
  I_c = np.identity(args.Jc)

  for category in range(args.C):
    
    beta = eta + (sz-1)/2
    #storing partial correlations
    P = np.zeros((sz,sz))
    #correlation matrix
    omega_c = np.eye(sz)

    for k in range(0, sz-1):
      beta = beta - 1/2
      for i in range(k+1, sz):
        #partial correlations from beta distribution
        P[k,i] = np.random.beta(0.2, 1)
        #linearly shifting to [-1, 1]
        #P[k,i] = (P[k,i]-0.5)*2;     
        p = P[k,i]
        #converting partial correlation to raw correlation
        for l in range(k-1,-1,-1):
            p = p * np.sqrt((1-P[l,i]**2)*(1-P[l,k]**2)) + P[l,i]*P[l,k]
        omega_c[k,i] = p
        omega_c[i,k] = p
    ##show the distribution of correlations
    #import matplotlib.pyplot as plt
    #_ = plt.hist(omega_c[np.triu_indices(sz,1)], bins=np.arange(0,1,0.1))  # arguments are passed to np.histogram
    #plt.show()
    tau_I_c = tau * I_c
    sigma_c = np.matmul(tau_I_c, omega_c)
    sigma_c = np.matmul(sigma_c, tau_I_c)
    Sigma_c[category, :, :] = sigma_c
  return Sigma_c


    


#products are selected by MNP model per category
#u_ijt = alpha_ij - beta * p_j + e_ijt
#alpha_ij ~ MVN(0, Sigma_c)
#Sigma_c = (tau * I_c) * omega_c * (tau * I_c)
#e_ijt ~ N(0, sigma0)
def genrate_utility(args):
  #Sigma_c = sigma_per_cat(args)
  Sigma_c = sigma_per_cat_vine(args)
  #base utility per person-week and product
  alpha_c_itj = np.empty((args.C, args.I, args.T, args.Jc))
  for category in range(args.C):
    alpha_c_itj[category, :, :, :] = np.random.multivariate_normal(
      mean = [0] * args.Jc, cov = Sigma_c[category, :, :], size = (args.I, args.T))
  #transpose week and product axis for proper order later
  alpha_c_ijt = alpha_c_itj.transpose(0,1,3,2)

  #random error term
  e_c_ijt = np.random.normal(0, args.Sigma0, size = (args.C, args.I, args.Jc, 
                             args.T))
  
  #price disutility
  p_c = np.random.lognormal(mean=0.5, sigma=0.3, size=args.C)
  p_c_j = np.random.uniform(low=p_c/2 , high=2*p_c, size=(args.Jc, args.C)).T
  
  #utility function
  utility_c_ijt = alpha_c_ijt - args.Beta * p_c_j[:,None,:,None] + e_c_ijt
  
  return utility_c_ijt, p_c_j

#select purchasing categries
def select_categories(args):
  Omega = omega()
  z = np.random.multivariate_normal(mean = [args.Gamma0] * args.C, 
                                    cov = Omega, size = (args.I, args.T))
  return z

 
#create master file for p2v code
def generate_master(args, p_c_j):
  with open(join_path('p2v', 'master.csv'), 'a') as p2v_master_file:
    p2v_master_file.write(',c,j,price\n')
    cnt = 0
    for i in range(args.C):
      for j in range(args.Jc):
        p2v_master_file.write(','.join(map(str, [cnt, i, cnt, p_c_j[i, j]])) + '\n')
        cnt += 1 

#generate baskets by choosing purchasing categories and selected products
#of each category and saving them into files

def data_generator(args):
  
  #select purchasing categries
  y = select_categories(args)
  print('catagories selected')

  #select products per purchasing categories
  utility_c_ijt, p_c_j = genrate_utility(args)
  print('utilities generated')

  #create data directory and delete content of files if already exist
  data_direcotry = 'data_graph'
  if not os.path.isdir(data_direcotry):
    os.mkdir(data_direcotry)
  if not os.path.isdir('p2v'):
    os.mkdir('p2v')
  
  #files to do basket completion
  open(join_path(data_direcotry, args.output + '.csv'), 'w').close()
  open(join_path(data_direcotry, args.output + '_train.csv'), 'w').close()
  open(join_path(data_direcotry, args.output + '_test.csv'), 'w').close()
  open(join_path(data_direcotry, args.output + '_validation.csv'), 'w').close()
  #files to get p2v embedding
  open(join_path('p2v', args.output + '.csv'), 'w').close()
  open(join_path('p2v', args.output + '_train.csv'), 'w').close()
  open(join_path('p2v', args.output + '_test.csv'), 'w').close()
  open(join_path('p2v', args.output + '_validation.csv'), 'w').close()
  open(join_path('p2v', 'master.csv'), 'w').close()

  #open files to write output to and add header line
  baskets_file = open(join_path(data_direcotry, args.output + '.csv'), 'a')
  baskets_file.write('basket_id,i,t,products\n')
  baskets_file_train = open(join_path(data_direcotry, args.output + '_train.csv'), 'a')
  baskets_file_train.write('products\n')
  baskets_file_test = open(join_path(data_direcotry, args.output + '_test.csv'), 'a')
  baskets_file_test.write('products\n')
  baskets_file_validation = open(join_path(data_direcotry, args.output + '_validation.csv'), 'a')
  baskets_file_validation.write('products\n')
  p2v_basket_file = open(join_path('p2v', args.output + '.csv'), 'a')
  p2v_basket_file.write(',i,j,price,price_paid,discount,t,basket_hash\n')
  p2v_basket_file_train = open(join_path('p2v', args.output + '_train.csv'), 'a')
  p2v_basket_file_train.write(',i,j,price,price_paid,discount,t,basket_hash\n')
  p2v_basket_file_test = open(join_path('p2v', args.output + '_test.csv'), 'a')
  p2v_basket_file_test.write(',i,j,price,price_paid,discount,t,basket_hash\n')
  p2v_basket_file_validation = open(join_path('p2v', args.output + '_validation.csv'), 'a')
  p2v_basket_file_validation.write(',i,j,price,price_paid,discount,t,basket_hash\n')
  generate_master(args, p_c_j)

  start = timer()
  #basket generation
#  baskets_it = []
  cnt = -1
  train_cnt = -1
  test_cnt = -1
  val_cnt = -1
  basket_id = -1

  data_split = list(map(float, args.data_split.split(',')))

  ###generate list of basket sizes from a fitted weibull distribution
  ###also adjust basket sizes to args.p (select two item per category)
  baskets_sz = np.round((1/(1+args.p)) * 
                        weibull_min.rvs(0.8046973517087279, 
                                        loc=2, 
                                        scale=1.4738173215687265, 
                                        size=args.I*args.T)).astype(int)

  #for each costumer
  for i in range(args.I):
#    baskets_i = []
    #for each week
    for t in range(args.T):
      basket = []
      basket_id += 1
      
      ###loop over categories that are selected to purchase from
      ###pick the categories above 0 -- p2v paper
      #for category in np.flatnonzero(y[i, t, :]>0):
      
      ###limit the selected categories to basket size
      basket_sz = min(baskets_sz[i * args.T + t] - 1, args.C - 1)
      for category in np.argpartition(-y[i, t, :], basket_sz)[:basket_sz+1]:
          cnt += 1
          #pick the product with highest utility
          idx = np.argmax(utility_c_ijt[category, i, :, t])
          j = category * args.Jc + idx
          basket.append(j)
          p2v_basket_file.write(','.join(map(str, 
            [cnt, i, j, 1, 1, 0, t, basket_id])) + '\n')
          #separate into train-test-validation sets
          #save each item in one line (p2v style)
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
          
          #add another item from the same category in p percent of times
          if np.random.uniform() < args.p:
            cnt += 1
            #pick the product with highest utility
            idx = np.argpartition(utility_c_ijt[category, i, :, t], -2)[-2]
            j = category * args.Jc + idx
            basket.append(j)
            p2v_basket_file.write(','.join(map(str, 
              [cnt, i, j, 1, 1, 0, t, basket_id])) + '\n')
            #separate into train-test-validation sets
            #save each item in one line (p2v style)
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
      #save each generated basket in one line
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
                      default = -0.5)#p2v paper set to -0.5
  parser.add_argument("-Sigma0", type = float, 
                      help = "Standard deviation for error term in MNP",
                      default = 1)
  parser.add_argument("-tau0", type = float, 
                      help = "Standard deviation for covariance matrices in MNP",
                      default = 2)
  parser.add_argument("-Beta", type = float, 
                      help = "Price sensitivity",
                      default = 2)
  parser.add_argument("-p", type = float, 
                      help = "Probability of sampling from same category twice",
                      default = 0.5)
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