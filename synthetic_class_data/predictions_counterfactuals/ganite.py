import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)

class GANITE:
    def __init__(self, X, T, Y, alpha=1.0, h_dim=8):
        self.X = X
        self.T = T.reshape(-1, 1)
        self.Y = Y.reshape(-1, 1)
        self.alpha = alpha
        self.h_dim = h_dim
        self.dim = X.shape[1]
        self.no = X.shape[0]
        
        # Build the model
        self._build_model()
        
    def _build_model(self):
        tf.reset_default_graph()
        
        # Placeholders
        self.X_ph = tf.placeholder(tf.float32, shape=[None, self.dim])
        self.T_ph = tf.placeholder(tf.float32, shape=[None, 1])
        self.Y_ph = tf.placeholder(tf.float32, shape=[None, 1])
        
        # Generator variables
        self.G_W1 = tf.Variable(xavier_init([(self.dim+2), self.h_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))
        self.G_W2 = tf.Variable(xavier_init([self.h_dim, self.h_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.h_dim]))
        self.G_W31 = tf.Variable(xavier_init([self.h_dim, self.h_dim]))
        self.G_b31 = tf.Variable(tf.zeros(shape=[self.h_dim]))
        self.G_W32 = tf.Variable(xavier_init([self.h_dim, 1]))
        self.G_b32 = tf.Variable(tf.zeros(shape=[1]))
        self.G_W41 = tf.Variable(xavier_init([self.h_dim, self.h_dim]))
        self.G_b41 = tf.Variable(tf.zeros(shape=[self.h_dim]))
        self.G_W42 = tf.Variable(xavier_init([self.h_dim, 1]))
        self.G_b42 = tf.Variable(tf.zeros(shape=[1]))
        
        self.theta_G = [self.G_W1, self.G_W2, self.G_W31, self.G_W32, self.G_W41, self.G_W42, 
                       self.G_b1, self.G_b2, self.G_b31, self.G_b32, self.G_b41, self.G_b42]
        
        # Discriminator variables
        self.D_W1 = tf.Variable(xavier_init([(self.dim+2), self.h_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))
        self.D_W2 = tf.Variable(xavier_init([self.h_dim, self.h_dim]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[self.h_dim]))
        self.D_W3 = tf.Variable(xavier_init([self.h_dim, 1]))
        self.D_b3 = tf.Variable(tf.zeros(shape=[1]))
        
        self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]
        
        # Build networks
        self.Y_tilde_logit = self._generator(self.X_ph, self.T_ph, self.Y_ph)
        self.Y_tilde = tf.nn.sigmoid(self.Y_tilde_logit)
        self.D_logit = self._discriminator(self.X_ph, self.T_ph, self.Y_ph, self.Y_tilde)
        
        # Loss functions
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.T_ph, logits=self.D_logit))
        self.G_loss_GAN = -self.D_loss
        self.G_loss_Factual = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.Y_ph, 
            logits=(self.T_ph * tf.reshape(self.Y_tilde_logit[:,1],[-1,1]) + 
                   (1. - self.T_ph) * tf.reshape(self.Y_tilde_logit[:,0],[-1,1]))))
        self.G_loss = self.G_loss_Factual + self.alpha * self.G_loss_GAN
        
        # Solvers
        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.theta_G)
        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.theta_D)
        
        # Session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def _generator(self, x, t, y):
        inputs = tf.concat(axis=1, values=[x, t, y])
        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        G_h31 = tf.nn.relu(tf.matmul(G_h2, self.G_W31) + self.G_b31)
        G_logit1 = tf.matmul(G_h31, self.G_W32) + self.G_b32
        G_h41 = tf.nn.relu(tf.matmul(G_h2, self.G_W41) + self.G_b41)
        G_logit2 = tf.matmul(G_h41, self.G_W42) + self.G_b42
        G_logit = tf.concat(axis=1, values=[G_logit1, G_logit2])
        return G_logit
    
    def _discriminator(self, x, t, y, hat_y):
        input0 = (1.-t) * y + t * tf.reshape(hat_y[:,0], [-1,1])
        input1 = t * y + (1.-t) * tf.reshape(hat_y[:,1], [-1,1])
        inputs = tf.concat(axis=1, values=[x, input0, input1])
        D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = tf.matmul(D_h2, self.D_W3) + self.D_b3
        return D_logit
    
    def train(self, iterations=10000):
        for it in range(iterations):
            # Train discriminator
            for _ in range(2):
                _, D_loss_curr = self.sess.run([self.D_solver, self.D_loss], 
                                             feed_dict={self.X_ph: self.X, 
                                                       self.T_ph: self.T, 
                                                       self.Y_ph: self.Y})
            
            # Train generator
            _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss], 
                                         feed_dict={self.X_ph: self.X, 
                                                   self.T_ph: self.T, 
                                                   self.Y_ph: self.Y})
    
    def generate_counterfactuals(self):
        Y_hat = self.sess.run(self.Y_tilde, feed_dict={self.X_ph: self.X, 
                                                      self.T_ph: self.T, 
                                                      self.Y_ph: self.Y})
        return Y_hat