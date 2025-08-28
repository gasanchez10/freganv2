import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from utils import xavier_init

def ganite(train_x, train_t, train_y, test_x, test_y, test_t, parameters):
  h_dim = parameters['h_dim']
  batch_size = parameters['batch_size']
  iterations = parameters['iteration']
  alpha = parameters['alpha']
  
  no, dim = train_x.shape
  tf.reset_default_graph()

  # Placeholders
  X = tf.placeholder(tf.float32, shape = [None, dim])
  T = tf.placeholder(tf.float32, shape = [None, 1])
  Y = tf.placeholder(tf.float32, shape = [None, 1])

  # Generator variables
  G_W1 = tf.Variable(xavier_init([(dim+2), h_dim]))
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  G_W31 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b31 = tf.Variable(tf.zeros(shape = [h_dim]))
  G_W32 = tf.Variable(xavier_init([h_dim, 1]))
  G_b32 = tf.Variable(tf.zeros(shape = [1]))
  G_W41 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b41 = tf.Variable(tf.zeros(shape = [h_dim]))
  G_W42 = tf.Variable(xavier_init([h_dim, 1]))
  G_b42 = tf.Variable(tf.zeros(shape = [1]))
  
  theta_G = [G_W1, G_W2, G_W31, G_W32, G_W41, G_W42, G_b1, G_b2, G_b31, G_b32, G_b41, G_b42]
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([(dim+2), h_dim]))
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  D_W3 = tf.Variable(xavier_init([h_dim, 1]))
  D_b3 = tf.Variable(tf.zeros(shape = [1]))
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

  # Generator
  def generator(x, t, y):
    inputs = tf.concat(axis = 1, values = [x,t,y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h31 = tf.nn.relu(tf.matmul(G_h2, G_W31) + G_b31)
    G_logit1 = tf.matmul(G_h31, G_W32) + G_b32
    G_h41 = tf.nn.relu(tf.matmul(G_h2, G_W41) + G_b41)
    G_logit2 = tf.matmul(G_h41, G_W42) + G_b42
    G_logit = tf.concat(axis = 1, values = [G_logit1, G_logit2])
    return G_logit
      
  # Discriminator
  def discriminator(x, t, y, hat_y):
    input0 = (1.-t) * y + t * tf.reshape(hat_y[:,0], [-1,1])
    input1 = t * y + (1.-t) * tf.reshape(hat_y[:,1], [-1,1])
    inputs = tf.concat(axis = 1, values = [x, input0,input1])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    return D_logit

  # Structure
  Y_tilde_logit = generator(X, T, Y)
  Y_tilde = tf.nn.sigmoid(Y_tilde_logit)
  D_logit = discriminator(X,T,Y,Y_tilde)

  # Loss functions
  D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = T, logits = D_logit))
  G_loss_GAN = -D_loss 
  G_loss_Factual = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      labels = Y, logits = (T * tf.reshape(Y_tilde_logit[:,1],[-1,1]) + 
                            (1. - T) * tf.reshape(Y_tilde_logit[:,0],[-1,1]))))
  G_loss = G_loss_Factual + alpha * G_loss_GAN

  # Solvers
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)

  # Training
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
      
  for it in range(iterations):
    for _ in range(2):
      _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {
          X: np.concatenate((train_x, test_x), axis=0),
          T: np.reshape(np.concatenate((train_t, test_t), axis=0), [len(train_t)+len(test_t),1]),
          Y: np.reshape(np.concatenate((train_y,test_y), axis=0), [len(train_t)+len(test_t),1])})
      
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {
        X: np.concatenate((train_x, test_x), axis=0),
        T: np.reshape(np.concatenate((train_t, test_t), axis=0), [len(train_t)+len(test_t),1]),
        Y: np.reshape(np.concatenate((train_y,test_y), axis=0), [len(train_t)+len(test_t),1])})
            
  # Generate potential outcomes
  test_y_hat = sess.run(Y_tilde, feed_dict = {
      X: np.concatenate((train_x, test_x), axis=0),
      T: np.reshape(np.concatenate((train_t, test_t), axis=0), [len(train_t)+len(test_t),1]),
      Y: np.reshape(np.concatenate((train_y,test_y), axis=0), [len(train_t)+len(test_t),1])})
  
  return test_y_hat