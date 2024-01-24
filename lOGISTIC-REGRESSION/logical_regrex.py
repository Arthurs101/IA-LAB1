import numpy as np
'''
X is goint to be a class
where has the next attributes
Shape : data, features

data: rows containing the data
features: the headers basically
it has to have a matrix with the data
'''

def sF(x):
    '''
    Represents the sigmoid function
    S(x) = 1/(1+e^[-x])
    '''
    Sx = 1/(1+np.exp(-x))
    return Sx

class logical_Rgrex:
    def __init__(self,L=0.01, n=2000):
        '''
        L: learining rate 
        n: number of iterations to do
        '''
        self.l_rate = L
        self.iterate = n 
        self.weights = None
        self.bias = None
    
    
    def fit_function(self,X,y):
        '''
        X: data as input
        y: the expected value for the Xi value
        Initializes de fitting of the model with bias as 0, weights as 0
        Sets the features and samples from the input data
        Function implied : y(x) = 1 / 1 + e^(-wx +b)
        '''

        samples,features = X.shape
        self.weights = np.zeros((features,1)) #No. of weights = No. Features
        self.bias = 0

        #start the training for the machine
        for _ in range(self.iterate):
            #calculates the wx + b 
            power = np.dot(X,self.weights) + self.bias 
            
            output = sF(power)
            #calculate the error and adjust the weights 'n bias accordingly 
            
            #the gradient for each one
            dw = 1/samples*np.dot(X.T , (output - y))
            db = 1/samples*np.sum(output - y)
            
            # ajsdusting value = value - learning rate * gradient
            self.weights = self.weights - self.l_rate*dw
            self.bias = self.bias - self.l_rate*db


    def predict(self, X):
        '''
        recieves data of a df, (X) outputs np.array with n amout, the same as samples
        '''
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sF(linear_pred)
        outputs = np.array([0 if y<=0.5 else 1 for y in y_pred])
        return outputs

    

        
        


