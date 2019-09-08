# %% 1
# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',  categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',  categories=categories)

num_train = len(newsgroups_train.data)
num_test  = len(newsgroups_test.data)

# max_features is an important parameter. You should adjust it.
vectorizer = TfidfVectorizer(max_features=65)

X = vectorizer.fit_transform( newsgroups_train.data + newsgroups_test.data )
X_train = X[0:num_train, :]
X_test = X[num_train:num_train+num_test,:]

Y_train = newsgroups_train.target
Y_test = newsgroups_test.target

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X_train, Y_train)

Y_predict = clf.predict(X_test)

print(Y_test)
print(Y_predict)

ncorrect = 0
for dy in  (Y_test - Y_predict):
	if 0 == dy:
		ncorrect += 1

print('text_logic classification accuracy is {}%'.format(round(100.0*ncorrect/len(Y_test)) ) )

class network(object):
    def __init__(self,num_ftrs=65,num_cl=4,lamda1=0.5,lamda2=0.5):
        self.num_ftrs=num_ftrs
        self.num_cl=num_cl
        self.lamda1=lamda1
        self.lamda2=lamda2
        
        W1=np.random.randn(num_ftrs,8)
        b1=np.random.randn(1,8)
        W2=np.random.randn(8,num_cl)
        b2=np.random.randn(1,num_cl)
        
        self.model={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
        
    def forward(self,X,y):
        self.num_example=X.shape[0]
        self.X=X
        self.y=y
        self.label=np.zeros((X.shape[0],self.num_cl))
        for i,j in enumerate(y):
            self.label[i,j]=1
        
        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        self.z1=X.dot(W1)+b1
        self.a1=np.tanh(self.z1)
        self.z2=self.a1.dot(W2)+b2
        exp_scores=np.exp(self.z2)
        self.probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
        
        y_pred=np.argmax(self.probs,axis=1)
        loss=-(self.label*np.log(self.probs)).sum()/X.shape[0]\
        +self.lamda1*(W1**2).sum()+self.lamda2*(W2**2).sum()
        
        return loss,y_pred
    
    def backward(self):
        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        delta3=self.probs
        delta3[range(self.num_example),self.y]-=1
        dW2=(self.a1.T).dot(delta3)+self.lamda2*W2
        db2=np.sum(delta3,axis=0,keepdims=True)
        delta2=delta3.dot(W2.T)*(1-np.power(self.a1,2))
        dW1=((self.X).T).dot(delta2)+self.lamda1*W1
        db1=np.sum(delta2,axis=0)
        
        return dW1,db1,dW2,db2
    
    def update(self,dX,learning_rate=0.0001):
        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        W1-=learning_rate*dX[0]
        b1-=learning_rate*dX[1]
        W2-=learning_rate*dX[2]
        b2-=learning_rate*dX[3]
        
        self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']=W1,b1,W2,b2
        
if __name__=='__main__':
    net=network()
    acc=0
    while acc<0.654:
        loss,y_pred=net.forward(X_train,Y_train)
        acc=(y_pred==Y_train).sum()/Y_train.shape[0]
        #print('loss:',loss,'acc:',acc)
        dW1,db1,dW2,db2=net.backward()
        net.update([dW1,db1,dW2,db2])
    
    loss,y_pred=net.forward(X_test,Y_test)  
    #print(y_pred[:20])
    ncorrect1 = 0
    for dy in  (Y_test - y_pred):
        if 0 == dy:
        	ncorrect1 += 1

    print('text_bp classification accuracy is {}%'.format(round(100.0*ncorrect/len(Y_test))))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        