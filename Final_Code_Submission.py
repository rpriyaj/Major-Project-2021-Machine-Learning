#IMPORTING LIBRARIES------------------------------------------------------------------
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import simpledialog as sd
from tkinter import messagebox as mb
from tkinter import ttk
import pandas as pd
import csv
from sklearn import svm
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from sklearn import metrics
import estimate_plot
import shap


#TKINTER GUI-----------------------------------------------------------------------
window = tk.Tk()
window.title("Sample GUI Application")

#resize window
window.geometry("600x200")

#Select file
selectfile = tk.Label(window, text="Choose File:").place(x=10, y=20)
selectfile_var = tk.StringVar()
selectfileEntryBox = tk.Entry(window, width=50, textvariable = selectfile_var)
selectfileEntryBox.place(x=120, y=20)
#Browse function
def browsefunc():
    global dataset
    global x_train
    global x_test
    global y_train
    global y_test
    global autoscaled_x_test
    global autoscaled_x_train
    global autoscaled_y_train
    filename = fd.askopenfilename(filetypes=(("csv files","*.csv"),("All files","*.*")))
    selectfileEntryBox.insert(tk.END, filename) 
    dataset = pd.read_csv(filename,index_col=-1)
    y_number = 6
    y = dataset.iloc[:, y_number].copy()
    x = dataset.iloc[:, :5]
    x = (x.T / x.T.sum()).T
    number_of_test_samples = 6
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,random_state=2)
    std_0_variable_flags = x_train.std() == 0
    x_train = x_train.drop(x_train.columns[std_0_variable_flags], axis=1)
    x_test = x_test.drop(x_test.columns[std_0_variable_flags], axis=1)

    #autoscaling
    autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
    autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
    autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()
    autoscaled_x_train = autoscaled_x_train.dropna()
    autoscaled_y_train = autoscaled_y_train.dropna()
    return dataset

    
#Browse Button
browsebtn=tk.Button(window,text="Browse",command=browsefunc).place(x=450,y=15)



#Defining all the regression methods
#1. Partial Least Squares Regression
def PartialLeastSquares(x,y,params):
    params = {k:int(p.get()) for k,p in params.items()}
    print(params)
    model = PLSRegression(**params)
    model.set_params(**params)
    return model.fit(x,y)
   
#2. Support Vector Regression
def SupportVectorRegression(x,y,params):
    params = {k:int(p.get()) for k,p in params.items()}
    print(params)
    model  = svm.SVR(**params)
    model.set_params(**params)
    return model.fit(x,y)

#3. Kernel Ridge Regression
def Kernelridge(x,y,params):
    params = {k:int(p.get()) for k,p in params.items()}
    print(params)
    model = KernelRidge(**params)
    model.set_params(**params)
    return model.fit(x,y)

#4. Random Forest Regression
def RandomForest(x,y,params):
    params = {k:int(p.get()) for k,p in params.items()}
    print(params)
    model = RandomForestRegressor(**params)
    model.set_params(**params)
    return model.fit(x,y)

#5. Adaptive Boosting Regression
def AdaBoost(x,y,params):
    params = {k:int(p.get()) for k,p in params.items()}
    print(params)
    model = AdaBoostRegressor(**params)
    model.set_params(**params)
    return model.fit(x,y)

#dropdown for users to Select Regression Method 
selectmthd = tk.Label(window, text ="Select Algorithm:").place(x=10, y=80)
options = ["Partial Least Squares Regression",
"Support Vector Regression",
"Kernel Ridge Regression",
"Random Forest Regression",
"Adaptive Boosting Regression"
]
selectmthdComboBox = ttk.Combobox(window, value=options)
selectmthdComboBox.place(x=120, y=80, width=180)


def clickoptions():
    modelname = selectmthdComboBox.get()
    if selectmthdComboBox.get() == "Partial Least Squares Regression":
        plsinputs=tk.Tk()
        plsinputs.title("Input Hyperparameters for Partial Least Squares Regression")
        plsinputs.geometry("300x200")
        plsncomp = tk.IntVar()
        plsmaxiter = tk.IntVar()
        pls1 = tk.Entry(plsinputs, width = 8,textvariable=plsncomp)
        pls2 = tk.Entry(plsinputs, width = 8,textvariable=plsmaxiter)
        pls1.place(x=120, y=20)
        pls2.place(x=120, y=60)
        plslabel1 = tk.Label(plsinputs, text="n_components:").place(x=10, y=20)
        plslabel2 = tk.Label(plsinputs, text="max_iter:").place(x=10, y=60)
        allparams = {"n_components":pls1,"max_iter":pls2}
        plsresultsbtn = tk.Button(plsinputs,text="Check Results",command = lambda: displayresults(modelname,allparams)).place(x=100, y=160)

        

    if selectmthdComboBox.get() == "Support Vector Regression":
        #GUI for SVR Input Hyperparameters window
        svrinputs = tk.Tk()
        svrinputs.title("Input Hyperparameters for Support Vector Regression")
        svrinputs.geometry("400x200")
        #Declare variables for hyperparameters
        svrdegree = tk.IntVar()
        svrcachesize = tk.IntVar()
        svrmaxiter = tk.IntVar()
        #Entry boxes for users to input hyperparameters
        svr1 = tk.Entry(svrinputs, width = 8, textvariable=svrdegree)
        svr2 = tk.Entry(svrinputs, width = 8, textvariable=svrcachesize)
        svr3 = tk.Entry(svrinputs, width = 8, textvariable=svrmaxiter)
        svr1.place(x=120, y=20)
        svr2.place(x=120, y=60)
        svr3.place(x=120, y=100)
        #Labels to display names of hyperparameters
        svrlabel1 = tk.Label(svrinputs, text="degree:").place(x=10, y=20)
        svrlabel2 = tk.Label(svrinputs, text="cache_size:").place(x=10, y=60)
        svrlabel3 = tk.Label(svrinputs, text="max_iter:").place(x=10, y=100)
        #Obtain stored values for the hyperparameters
        allparams = {"degree":svr1, "cache_size":svr2, "max_iter":svr3}
        svrresultsbtn = tk.Button(svrinputs, text="Check Results", command = lambda: displayresults(modelname, allparams)).place(x=120, y=160)
        
    if selectmthdComboBox.get() == "Kernel Ridge Regression":
        #GUI for KR Input Hyperparameters window
        krinputs=tk.Tk()
        krinputs.title("Input Hyperparameters for Kernel Ridge Regression")
        krinputs.geometry("250x230")
        #Declare variables for hyperparameters
        kralpha = tk.IntVar()
        krdegree = tk.IntVar()
        #Entry boxes for users to input hyperparameters
        kr1 = tk.Entry(krinputs, width = 8, textvariable=kralpha)
        kr2 = tk.Entry(krinputs, width = 8, textvariable=krdegree)
        kr1.place(x=120, y=20)
        kr2.place(x=120, y=60)
        #Labels to display names of hyperparameters
        krlabel1 = tk.Label(krinputs, text="alpha:").place(x=10, y=20)
        krlabel2 = tk.Label(krinputs, text="degree:").place(x=10, y=60)
        #Obtain stored values for the hyperparameters
        allparams = {"alpha":kr1, "degree":kr2}
        krresultsbtn = tk.Button(krinputs, text="Check Results", command = lambda: displayresults(modelname, allparams)).place(x=80, y=180)

    if selectmthdComboBox.get() == "Random Forest Regression":
        #GUI for RFR Input Hyperparameters window
        rfinputs = tk.Tk()
        rfinputs.title("Input Hyperparameters for Random Forest Regression")
        rfinputs.geometry("300x250")
        #Declare variables for hyperparameters
        rfnestimators = tk.IntVar()
        rfmaxdepth = tk.IntVar()
        rfminsamplessplit = tk.IntVar()
        rfminsamplesleaf = tk.IntVar()
        #Entry boxes for users to input hyperparameters
        rf1 = tk.Entry(rfinputs, width = 8, textvariable=rfnestimators)
        rf2 = tk.Entry(rfinputs, width = 8, textvariable=rfmaxdepth)
        rf3 = tk.Entry(rfinputs, width = 8, textvariable=rfminsamplessplit)
        rf4 = tk.Entry(rfinputs, width = 8, textvariable=rfminsamplesleaf)
        rf1.place(x=120, y=20)
        rf2.place(x=120, y=60)
        rf3.place(x=120, y=100)
        rf4.place(x=120, y=140)
        #Labels to display names of hyperparameters
        rflabel1 = tk.Label(rfinputs, text="n_estimators:").place(x=10, y=20)
        rflabel2 = tk.Label(rfinputs, text="max_depth:").place(x=10, y=60)
        rflabel3 = tk.Label(rfinputs, text="min_samples_split:").place(x=10, y=100)
        rflabel4 = tk.Label(rfinputs, text="min_samples_leaf:").place(x=10, y=140)
        #Obtain stored values for the hyperparameters
        allparams = {"n_estimators":rf1, "max_depth": rf2, "min_samples_split":rf3, "min_samples_leaf":rf4}
        rfresultsbtn = tk.Button(rfinputs, text="Check Results", command = lambda: displayresults(modelname, allparams)).place(x=100, y=190)
    
    if selectmthdComboBox.get() == "Adaptive Boosting Regression":
        #GUI for AB Input Hyperparameters window
        adaboostinputs=tk.Tk()
        adaboostinputs.title("Input Hyperparameters for Adaptive Boosting Regression")
        adaboostinputs.geometry("250x230")
        #Declare variables for hyperparameters
        abnestimators = tk.IntVar()
        ablearningrate = tk.IntVar()
        #Entry boxes for users to input hyperparameters
        ab1 = tk.Entry(adaboostinputs, width = 8, textvariable=abnestimators)
        ab2 = tk.Entry(adaboostinputs, width = 8, textvariable=ablearningrate)
        ab1.place(x=120, y=20)
        ab2.place(x=120, y=60)
        #Labels to display names of hyperparameters
        ablabel1 = tk.Label(adaboostinputs, text="n_estimators:").place(x=10, y=20)
        ablabel2 = tk.Label(adaboostinputs, text="learning_rate:").place(x=10, y=60)
        allparams = {"n_estimators":ab1, "learning_rate":ab2}
        abresultsbtn = tk.Button(adaboostinputs, text="Check Results", command = lambda : displayresults(modelname, allparams)).place(x=80, y=180)
submitButton = tk.Button(window, text="Submit", command=clickoptions).place(x=250, y=150)

#Store all the models in a dictionary
models = {"Partial Least Squares Regression":PartialLeastSquares,
"Support Vector Regression": SupportVectorRegression, "Kernel Ridge Regression": Kernelridge, "Random Forest Regression": RandomForest, "Adaptive Boosting Regression": AdaBoost}

def displayresults(modelname,allparams):
    print("display results - modelname = {}".format(modelname))
    print("display results - allparams = {}".format(allparams))
    model = models[modelname](autoscaled_x_train, autoscaled_y_train,allparams)
    estimate_plot.estimation_and_performance_check_in_regression_train_and_test(model,
     autoscaled_x_train, y_train, autoscaled_x_test, y_test)
    shap.initjs()
    explainer = shap.KernelExplainer(model.predict, autoscaled_x_train)
    shap_values = explainer.shap_values(autoscaled_x_train)
    shap.summary_plot(shap_values, autoscaled_x_train, plot_type="bar")
    shap.summary_plot(shap_values, autoscaled_x_train)


tk.mainloop()