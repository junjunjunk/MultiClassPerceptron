import numpy as np
from sklearn.utils import shuffle

class MultiClassPerceptron:

    def __init__(self, nclasses , in_dim): 
        self.weights = np.array( [np.zeros(in_dim+1).astype('float64') for i in range (nclasses) ] )
        # weight vector

        self.weights_mean = self.weights.copy()
        # mean of weight vectors


    def train(self,x_train,y_train,nepoch):
        # x:input_tensors t:labels
        # have to return mean of weights , so this training uses averaging perceptron.
        # training of averaging perceptron returns mean of weights from beginning of training to end of training.
        print(x_train.shape)
        feature_vectors = np.insert(x_train,0,1,axis=1)
        print(feature_vectors)
        #bias initialized 1

        for i in range(nepoch):
            feature_vectors, y_train = shuffle(feature_vectors, y_train )
            for x,t in zip(feature_vectors,y_train):

                output = np.matmul(self.weights,x.T)
                pred_y = np.argmax(output)

                if t==pred_y: 
                    pass
                else:
                    self.weights[t,:] += x
                    self.weights[pred_y,:] -= x

                self.weights_mean += self.weights
                print(pred_y,t,'\n',self.weights)

        self.weights = self.weights_mean/(nepoch*x_train.shape[0])
        # training times = epochs*x_train.shape[0]
        print(self.weights)


    def predict(self,x):
        feature_vector = np.insert(x,0,1)
        output = np.matmul(self.weights,feature_vector.T)
        pred_y = np.argmax(output)

        return pred_y


#below is example of train & predict (AND)

p = MultiClassPerceptron(2,2)

train = np.array([[0,0],[0,1],[1,0],[1,1]])
label = [0,0,0,1]
nepoch = 100

p.train(train,label,nepoch)

for x in train:
    print(p.predict(x))

