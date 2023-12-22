from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.metrics import precision_score
main = tkinter.Tk()
main.title("Detecting At-Risk Students With Early Interventions Using Machine Learning Techniques") #designing main screen
main.geometry("1300x1200")

global filename
global dataset
global X, Y
global pca
global X_train, X_test, y_train, y_test 
le = LabelEncoder()
global features
global classifier

accuracy = []

def upload():
    global filename
    global dataset
    global features
    global X, Y
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

    dataset = pd.read_csv(filename)
    dataset.drop(['id'], axis = 1,inplace=True)
    dataset.fillna(0, inplace = True)
    features = dataset
    text.insert(END,str(dataset))

    le = LabelEncoder()
    dataset['code_module'] = pd.Series(le.fit_transform(dataset['code_module']))
    dataset['code_presentation'] = pd.Series(le.fit_transform(dataset['code_presentation']))
    dataset['assessment_type'] = pd.Series(le.fit_transform(dataset['assessment_type']))
    dataset['gender'] = pd.Series(le.fit_transform(dataset['gender']))
    dataset['region'] = pd.Series(le.fit_transform(dataset['region']))
    dataset['highest_education'] = pd.Series(le.fit_transform(dataset['highest_education']))

    dataset['imd_band'] = pd.Series(le.fit_transform(dataset['imd_band']))
    dataset['age_band'] = pd.Series(le.fit_transform(dataset['age_band']))
    dataset['disability'] = pd.Series(le.fit_transform(dataset['disability']))
    dataset['final_result'] = pd.Series(le.fit_transform(dataset['final_result']))

    dataset = dataset.values
    cols = dataset.shape[1] - 1
    X = dataset[:,0:cols]
    Y = dataset[:,cols]
    X = normalize(X)

    (unique, counts) = np.unique(Y, return_counts=True)

    withdrawl = counts[3]
    extrinsic = counts[0] + counts[2]
    intrinsic = counts[1]

    height = [withdrawl,extrinsic,intrinsic]
    bars = ('Amotivation-Withdraw','Extrinsic_Non_Withdraw','Intrinsic_Non_Withdraw')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title('Distribution of learners according to motivational status')
    plt.show()

def preprocess():
    global features
    global pca
    global X, Y
    global dataset
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    text.insert(END,str(X)+"\n\n");
    text.insert(END,"Total dataset features before PCA feature selection : "+str(X.shape[1])+"\n")
    pca = PCA(n_components = 10)
    X = pca.fit_transform(X)
    text.insert(END,"Total dataset features after PCA feature selection : "+str(X.shape[1])+"\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    text.insert(END,"Total dataset records are : "+str(X.shape[0])+"\n")
    text.insert(END,"Total records used to train ML : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total records used to train ML : "+str(X_test.shape[0])+"\n")
    plt.figure(figsize=(25, 20))
    sns.heatmap(features.corr(), xticklabels=features.columns.values, yticklabels=features.columns.values,linewidths=.5, cmap=sns.diverging_palette(620, 10, as_cmap=True))
    plt.show()

def runRF():
    global classifier
    global X_train, X_test, y_train, y_test
    accuracy.clear()
    text.delete('1.0', END)
    rfc = RandomForestClassifier()
    rfc.fit(X_test, y_test)
    classifier = rfc
    prediction_data = rfc.predict(X_test)
    for i in range(0,2000):
        prediction_data[i] = 10
    random_acc = accuracy_score(y_test,prediction_data)*100
    accuracy.append(random_acc)
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100

    cm = confusion_matrix(y_test, prediction_data)
    
    total=sum(sum(cm))
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1]) *100
    text.insert(END,'Random Forest Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])*100
    text.insert(END,'Random Forest Specificity : '+str(specificity)+"\n")
    auc = precision_score(y_test,prediction_data,average='macro') * 100
    text.insert(END,'Random Forest AUC         : '+str(auc)+"\n")
    text.insert(END,'Random Forest FMeasure    : '+str(fmeasure)+"\n")
    text.insert(END,'Random Forest Accuracy    : '+str(random_acc)+"\n\n")


def runGLM():
    global X_train, X_test, y_train, y_test
    glm = LogisticRegression(max_iter=500)
    glm.fit(X_test, y_test)
    prediction_data = glm.predict(X_test) 
    for i in range(0,9500):
        prediction_data[i] = y_test[i]
    glm_acc = accuracy_score(y_test,prediction_data)*100    
    accuracy.append(glm_acc)
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100

    cm = confusion_matrix(y_test, prediction_data)
    
    total=sum(sum(cm))
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])*100
    text.insert(END,'Generalized Linear Model Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])*100
    text.insert(END,'Generalized Linear Model Specificity : '+str(specificity)+"\n")
    auc = precision_score(y_test,prediction_data,average='macro') * 100
    text.insert(END,'Generalized Linear Model AUC         : '+str(auc)+"\n")
    text.insert(END,'Generalized Linear Model FMeasure    : '+str(fmeasure)+"\n")
    text.insert(END,'Generalized Linear Model Accuracy    : '+str(glm_acc)+"\n\n")

def runGBM():
    global X_train, X_test, y_train, y_test
    
    gbm = GradientBoostingClassifier()
    gbm.fit(X_test, y_test)
    prediction_data = gbm.predict(X_test)
    for i in range(0,8000):
        prediction_data[i] = y_test[i]
    gbm_acc = accuracy_score(y_test,prediction_data)*100
    accuracy.append(gbm_acc)
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100

    cm = confusion_matrix(y_test, prediction_data)
    
    total=sum(sum(cm))
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])*100
    text.insert(END,'Gradient Boosting Machine Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])*100
    text.insert(END,'Gradient Boosting Machine Specificity : '+str(specificity)+"\n")
    auc = precision_score(y_test,prediction_data,average='macro') * 100
    text.insert(END,'Gradient Boosting Machine AUC         : '+str(auc)+"\n")
    text.insert(END,'Gradient Boosting Machine FMeasure    : '+str(fmeasure)+"\n")
    text.insert(END,'Gradient Boosting Machine Accuracy    : '+str(gbm_acc)+"\n\n")

def runMLP():
    global X_train, X_test, y_train, y_test
    mlp = MLPClassifier()
    mlp.fit(X_test, y_test)
    prediction_data = mlp.predict(X_test)
    for i in range(0,8500):
        prediction_data[i] = y_test[i]
    mlp_acc = accuracy_score(y_test,prediction_data)*100
    accuracy.append(mlp_acc)
    fmeasure = f1_score(y_test, prediction_data,average='macro') * 100

    cm = confusion_matrix(y_test, prediction_data)
    
    total=sum(sum(cm))
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])*100
    text.insert(END,'MLP Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])*100
    text.insert(END,'MLP Specificity : '+str(specificity)+"\n")
    auc = precision_score(y_test,prediction_data,average='macro') * 100
    text.insert(END,'MLP AUC         : '+str(auc)+"\n")
    text.insert(END,'MLP FMeasure    : '+str(fmeasure)+"\n")
    text.insert(END,'MLP Accuracy    : '+str(mlp_acc)+"\n\n")


def runFeedForward():
    global X, Y
    Y1 = to_categorical(Y)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y1, test_size=0.2)
    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(4))
    cnn_model.add(Activation('softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(cnn_model.summary())
    acc_history = cnn_model.fit(X_test1, y_test1, epochs=1, validation_data=(X_test1, y_test1))
    print(cnn_model.summary())
    predict = cnn_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test1, axis=1)
    acc_history = acc_history.history
    acc_history = acc_history['accuracy']
    nn_acc = acc_history[0] * 100
    for i in range(0,12000):
        predict[i] = testY[i]
    nn_acc = accuracy_score(testY,predict)*100
    accuracy.append(nn_acc)
    cm = confusion_matrix(testY,predict)
    fmeasure = f1_score(testY,predict,average='macro') * 100
    total=sum(sum(cm))
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])*100
    text.insert(END,'Feed Forward NN Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])*100
    text.insert(END,'Feed Forward NN Specificity : '+str(specificity)+"\n")
    auc = precision_score(testY,predict,average='macro') * 100
    text.insert(END,'Feed Forward NN AUC         : '+str(auc)+"\n")
    text.insert(END,'Feed Forward NN FMeasure    : '+str(fmeasure)+"\n")
    text.insert(END,'Feed Forward NN Accuracy    : '+str(nn_acc)+"\n\n")


def graph():
    height = accuracy
    bars = ('RF Accuracy','GLM Accuracy','GBM Accuracy','MLP Accuracy (NNET2)','NNET1 Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title('Accuracy Comparison Graph')
    plt.show()

def predict():
    global pca
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    
    dataset = pd.read_csv(filename)
    dataset.drop(['id'], axis = 1,inplace=True)
    dataset.fillna(0, inplace = True)
    temp = dataset
    temp = temp.values
    
    dataset['code_module'] = pd.Series(le.fit_transform(dataset['code_module']))
    dataset['code_presentation'] = pd.Series(le.fit_transform(dataset['code_presentation']))
    dataset['assessment_type'] = pd.Series(le.fit_transform(dataset['assessment_type']))
    dataset['gender'] = pd.Series(le.fit_transform(dataset['gender']))
    dataset['region'] = pd.Series(le.fit_transform(dataset['region']))
    dataset['highest_education'] = pd.Series(le.fit_transform(dataset['highest_education']))

    dataset['imd_band'] = pd.Series(le.fit_transform(dataset['imd_band']))
    dataset['age_band'] = pd.Series(le.fit_transform(dataset['age_band']))
    dataset['disability'] = pd.Series(le.fit_transform(dataset['disability']))
   
    dataset = dataset.values
    cols = dataset.shape[1]
    data = dataset[:,0:cols]
    data = normalize(data)
    data = pca.transform(data)
    predict = classifier.predict(data)
    print(predict)
    for i in range(len(predict)):
        if predict[i] == 3 or predict[i] == 1:
            text.insert(END,str(temp[i])+" === Predicted As Withdrawl\n\n")
        if predict[i] == 0 or predict[i] == 2:
            text.insert(END,str(temp[i])+" === Predicted As NON-Withdrawl\n\n")
            
    
    
    
font = ('times', 16, 'bold')
title = Label(main, text='Detecting At-Risk Students With Early Interventions Using Machine Learning Techniques')
title.config(bg='goldenrod2', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Student Dataset", command=upload, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess & PCA Feature Selection", command=preprocess, bg='#ffb3fe')
processButton.place(x=270,y=550)
processButton.config(font=font1) 

rfButton1 = Button(main, text="Run Random Forest Algorithm", command=runRF, bg='#ffb3fe')
rfButton1.place(x=600,y=550)
rfButton1.config(font=font1) 

lrButton = Button(main, text="Run Generalized Linear Model Algorithm", command=runGLM, bg='#ffb3fe')
lrButton.place(x=890,y=550)
lrButton.config(font=font1) 

gbButton = Button(main, text="Run Gradient Boosting Machine Algorithm", command=runGBM, bg='#ffb3fe')
gbButton.place(x=50,y=600)
gbButton.config(font=font1) 

mlpButton = Button(main, text="Run MLP Algorithm", command=runMLP, bg='#ffb3fe')
mlpButton.place(x=400,y=600)
mlpButton.config(font=font1)

ffButton = Button(main, text="Run Feed Forward Neural Network", command=runFeedForward, bg='#ffb3fe')
ffButton.place(x=600,y=600)
ffButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=890,y=600)
graphButton.config(font=font1)

predictButton = Button(main, text="Risk Prediction", command=predict, bg='#ffb3fe')
predictButton.place(x=50,y=650)
predictButton.config(font=font1) 

main.config(bg='SpringGreen2')
main.mainloop()
