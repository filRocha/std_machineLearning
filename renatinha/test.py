import numpy as np  # Note: there is a typo on this line in the video
import os
import pandas as pd


# Sigmoid creation
def nonlin(x, deriv=False):  # Note: there is a typo on this line in the video
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))  # Note: there is a typo on this line in the video

# ---- Importing xslx data

if __name__ == '__main__':

    cwd = os.getcwd()
    file = 'escor.xlsx'
    training_data = pd.read_excel(cwd+"/"+file, header=None)

    training_data = training_data.as_matrix()

    #print(training_data)

    X = training_data[:,[0,1,2,3,4]]
    Y = training_data[:,5]
    Y = np.array(Y)
    Y.shape = (116,1)

    # print(X)
    # print('---')
    # print(Y)

    # #input data
    # X = np.array([[0,0,1],  # Note: there is a typo on this line in the video
    #               [0,1,1],
    #               [1,0,1],
    #               [1,1,1]])
    #
    #
    #
    # #output data
    # y = np.array([[0],
    #               [1],
    #               [1],
    #               [0]])


    # The seed for the random generator is set so that it will return the same random numbers each time, which is sometimes useful for debugging.
    np.random.seed(1)


    # Now we intialize the weights to random values. syn0 are the weights between the input layer and the hidden layer.  It is a 3x4 matrix because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=4). syn1 are the weights between the hidden layer and the output layer. It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output. Note that there is no bias term feeding the output layer in this example. The weights are initially generated randomly because optimization tends not to work well when all the weights start at the same value. Note that neither of the neural networks shown in the video describe the example.

    # In[28]:

    #synapses
    syn0 = 2*np.random.random((5,50)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
    syn1 = 2*np.random.random((50,1)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.


    # This is the main training loop. The output shows the evolution of the error between the model and desired. The error steadily decreases.

    # In[29]:

    #training step
    # Python2 Note: In the follow command, you may improve
    #   performance by replacing 'range' with 'xrange'.
    for j in range(60000):

        # Calculate forward through the network.
        l0 = X
        l1 = nonlin(np.dot(l0, syn0))
        l2 = nonlin(np.dot(l1, syn1))

        # Back propagation of errors using the chain rule.
        l2_error = Y - l2
        if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output.
            print("Error: " + str(np.mean(np.abs(l2_error))))

        l2_delta = l2_error*nonlin(l2, deriv=True)

        l1_error = l2_delta.dot(syn1.T)

        l1_delta = l1_error * nonlin(l1,deriv=True)

        #update weights (no learning rate term)
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    print("Output after training")
    print(l2)




    # See how the final output closely approximates the true output [0, 1, 1, 0]. If you increase the number of interations in the training loop (currently 60000), the final output will be even closer.

    # In[30]:


# In[ ]:



