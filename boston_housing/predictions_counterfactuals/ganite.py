import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from utils import xavier_init


def ganite(train_x, train_t, train_y, test_x, test_y, test_t, parameters):
    """GANITE: Generative Adversarial Nets for Individualized Treatment Effects.
    
    Estimates counterfactual outcomes using adversarial training between
    a generator and discriminator network.
    """
    # Extract hyperparameters
    h_dim = parameters['h_dim']        # Hidden layer dimensions
    iterations = parameters['iteration'] # Training iterations
    alpha = parameters['alpha']        # Loss balance parameter
    
    no, dim = train_x.shape
    tf.reset_default_graph()

    # Input placeholders
    X = tf.placeholder(tf.float32, shape=[None, dim])  # Features
    T = tf.placeholder(tf.float32, shape=[None, 1])    # Treatment assignment
    Y = tf.placeholder(tf.float32, shape=[None, 1])    # Observed outcomes

    # Generator network parameters (input: X + T + Y -> potential outcomes)
    G_W1 = tf.Variable(xavier_init([(dim+2), h_dim]))  # Input layer weights
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))     # Hidden layer weights
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    
    # Multi-task output heads for T=0 and T=1 outcomes
    G_W31 = tf.Variable(xavier_init([h_dim, h_dim]))    # T=0 outcome head
    G_b31 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W32 = tf.Variable(xavier_init([h_dim, 1]))
    G_b32 = tf.Variable(tf.zeros(shape=[1]))
    G_W41 = tf.Variable(xavier_init([h_dim, h_dim]))    # T=1 outcome head
    G_b41 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W42 = tf.Variable(xavier_init([h_dim, 1]))
    G_b42 = tf.Variable(tf.zeros(shape=[1]))
    
    theta_G = [G_W1, G_W2, G_W31, G_W32, G_W41, G_W42, G_b1, G_b2, G_b31, G_b32, G_b41, G_b42]
  
    # Discriminator network parameters (input: X + factual + counterfactual -> treatment prediction)
    D_W1 = tf.Variable(xavier_init([(dim+2), h_dim]))  # Input layer weights
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))     # Hidden layer weights
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W3 = tf.Variable(xavier_init([h_dim, 1]))        # Output layer weights
    D_b3 = tf.Variable(tf.zeros(shape=[1]))
    
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  


    def generator(x, t, y):
        """Generate potential outcomes for both treatment conditions."""
        # Concatenate features, treatment, and observed outcome
        inputs = tf.concat(axis=1, values=[x, t, y])
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        
        # Generate outcome for T=0 (control)
        G_h31 = tf.nn.relu(tf.matmul(G_h2, G_W31) + G_b31)
        G_logit1 = tf.matmul(G_h31, G_W32) + G_b32
        
        # Generate outcome for T=1 (treatment)
        G_h41 = tf.nn.relu(tf.matmul(G_h2, G_W41) + G_b41)
        G_logit2 = tf.matmul(G_h41, G_W42) + G_b42
        
        return tf.concat(axis=1, values=[G_logit1, G_logit2])
    
    def discriminator(x, t, y, hat_y):
        """Discriminate between real and generated treatment assignments."""
        # Combine factual and counterfactual outcomes based on treatment
        input0 = (1.-t) * y + t * tf.reshape(hat_y[:,0], [-1,1])  # Control outcome
        input1 = t * y + (1.-t) * tf.reshape(hat_y[:,1], [-1,1])  # Treatment outcome
        inputs = tf.concat(axis=1, values=[x, input0, input1])
        
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        return tf.matmul(D_h2, D_W3) + D_b3
    


    # Generate potential outcomes and discriminator predictions
    Y_tilde = generator(X, T, Y)  # Generated potential outcomes
    D_logit = discriminator(X, T, Y, Y_tilde)  # Treatment prediction

    # Loss functions
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=T, logits=D_logit))
    
    G_loss_GAN = -D_loss  # Adversarial loss (fool discriminator)
    # Factual loss (match observed outcomes)
    G_loss_Factual = tf.reduce_mean(tf.squared_difference(Y, (T * tf.reshape(Y_tilde[:,1],[-1,1]) + 
                                                             (1. - T) * tf.reshape(Y_tilde[:,0],[-1,1]))))
    G_loss = G_loss_Factual + alpha * G_loss_GAN  # Combined generator loss

    # Optimizers
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)

    # Training session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Prepare combined training data
    X_all = np.concatenate((train_x.values, test_x.values), axis=0).astype(np.float32)
    T_all = np.reshape(np.concatenate((train_t.values, test_t.values), axis=0), [len(train_t)+len(test_t), 1]).astype(np.float32)
    Y_all = np.reshape(np.concatenate((train_y.values, test_y.values), axis=0), [len(train_t)+len(test_t), 1]).astype(np.float32)
    
    # Adversarial training loop
    for it in range(iterations):
        # Train discriminator twice per generator update
        for _ in range(2):
            sess.run(D_solver, feed_dict={X: X_all, T: T_all, Y: Y_all})
        
        # Train generator once
        sess.run(G_solver, feed_dict={X: X_all, T: T_all, Y: Y_all})
    
    # Generate counterfactual outcomes for test set
    test_y_hat=sess.run(Y_tilde, feed_dict = {X: X_all, T: T_all, Y: Y_all})
  
    sess.close()
    
    return test_y_hat        