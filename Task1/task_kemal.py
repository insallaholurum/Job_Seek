#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

class Learn():
    def __init__(self,data,excessive,categorical,numerical,target):
        self.data = data
        self.excessive = excessive
        self.categorical = categorical
        self.numerical = numerical
        self.target = target
        self.Run()
    def Clear(self):
        #drop excessive
        self.data.drop(columns=self.excessive,inplace=True)
        self.data.dropna(inplace=True)
        for i in numerical:
            self.data[i] = self.data[i].apply(int)
        self.data.reset_index(inplace=True,drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
    def Preprocess(self):
        #OneHotEncode
        self.encoder = OneHotEncoder(sparse=False)
        self.input = self.encoder.fit_transform(self.data[self.categorical])
        self.output = self.encoder.fit_transform(self.data[self.target])
        #Normalize
        self.min_max = []
        for i in numerical:
            x = self.data[i]
            self.min_max.append((min(x),max(x)))
            normalized = (x-min(x))/(max(x)-min(x))
            self.data[i] = normalized
        self.Normalized = np.array(self.data[numerical])
        self.Preprocess = np.concatenate((self.input,self.Normalized),axis=1)
        #concat
    def Reconstruct(self):
        #Ohe rec
        categories = [j for sub in self.encoder.categories_ for j in sub]
        self.categorical.extend(self.numerical)
        self.reconstructed = pd.DataFrame(columns= self.categorical)
        k = 0
        for i in self.Preprocess:
            i = list(i)
            for j in range(len(i)-2):
                if (i[j] == 1):
                    i[j] = categories[j] 
            self.reconstructed.loc[k,:] = element
            k+=1
        for i in range(len(self.numerical)):
            Normalized = np.array(self.data[self.numerical[i]])
            min_x = self.min_max[i][0]
            max_x = self.min_max[i][1]
            x = (Normalized*(max_x-min_x))+min_x
            self.reconstructed[self.numerical[i]] = x
    def Split(self):
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.Preprocess,self.output,test_size = 0.2)
    def Model(self):
        model = Sequential()
        model.add(Dense(8, input_dim=len(self.Preprocess[0]), activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = model.fit(self.x_train, self.y_train, epochs=100, batch_size=64)
        y_pred = model.predict(self.x_test)
        #Converting predictions to label
        pred = list()
        for i in range(len(y_pred)):
            pred.append(np.argmax(y_pred[i]))
        #Converting one hot encoded test label to label
        test = list()
        for i in range(len(self.y_test)):
            test.append(np.argmax(self.y_test[i]))
        a = accuracy_score(pred,test)
        print('Accuracy is:', a*100)
    def Plot(self):
        plt.plot(self.history.history['accuracy'])
        #plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    def Run(self):
        self.Clear()
        self.Preprocess()
        self.Split()
        self.Model()
        self.Plot()
        #self.Reconstruct()
#data inpıt
train = pd.read_csv('aug_train.csv')
train.loc[train['experience']== '>20',['experience']] = 20
train.loc[train['experience']== '<1',['experience']] = 0
#categorize columns
excessive = ['gender','enrollee_id','city','city_development_index']
categorical = ['relevent_experience','enrolled_university','education_level','major_discipline','company_size','company_type','last_new_job']
numerical = ['training_hours','experience']
target = ['target']
#run
test = Learn(train,excessive,categorical,numerical,target)
#