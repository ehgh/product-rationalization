import os
import argparse
import numpy as np
from scipy.linalg import block_diag
from timeit import default_timer as timer
from os.path import join as join_path

np.set_printoptions(precision = 1, suppress = True)

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
  return block_diag(D1, D2, D3, D4, D5, D6)


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
    omega_c = np.dot(b, b.T)
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
  Sigma_c = sigma_per_cat(args)
  #base utility per person and product
  alpha_c_ij = np.empty((args.C, args.I, args.Jc))
  for category in range(args.C):
    alpha_c_ij[category, :, :] = np.random.multivariate_normal(
      mean = [0] * args.Jc, cov = Sigma_c[category, :, :], size = (args.I))
  #repeat it over different weeks for faster array addition later
  alpha_c_ijt = np.repeat(alpha_c_ij[:, :, :, np.newaxis], args.T, axis = 3)

  e_c_ijt = np.random.normal(0, args.Sigma0, size = (args.C, args.I, args.Jc, 
                             args.T))
  #utility function
  utility_c_ijt = alpha_c_ijt + e_c_ijt

  return utility_c_ijt

#select purchasing categries
def select_categories(args):
  Omega = omega()
  z = np.random.multivariate_normal(mean = [args.Gamma0] * args.C, 
                                    cov = Omega, size = (args.I, args.T))
  return z > 0

  
#generate baskets by choosing purchasing categories and selected products
#of each category and saving them into files

#TO DO:
#  Add price sensitivity to utility function
#  Add selecting more products from a single category

def data_generator(args):
  
  #select purchasing categries
  y = select_categories(args)

  #select products per purchasing categories
  utility_c_ijt = genrate_utility(args)

  #create data directory and delete content of files if already exist
  data_direcotry = 'data'
  if not os.path.isdir(data_direcotry):
    os.mkdir(data_direcotry)
  if not os.path.isdir('p2v'):
    os.mkdir('p2v')
  open(join_path(data_direcotry, args.output + '.csv'), 'w').close()
  open(join_path(data_direcotry, args.output + '_train.csv'), 'w').close()
  open(join_path(data_direcotry, args.output + '_test.csv'), 'w').close()
  open(join_path(data_direcotry, args.output + '_validation.csv'), 'w').close()
  open(join_path('p2v', args.output + '.csv'), 'w').close()
  open(join_path('p2v', args.output + '_train.csv'), 'w').close()
  open(join_path('p2v', args.output + '_test.csv'), 'w').close()
  open(join_path('p2v', args.output + '_validation.csv'), 'w').close()

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


  start = timer()
  #basket generation
#  baskets_it = []
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