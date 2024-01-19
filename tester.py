import numpy as np
import pandas as pd
from logical_regrex import logical_Rgrex as IA
import csv 

#read the data using dataframes
main_data = pd.read_csv("dataset_phishing.csv")
#calculate the index for training and testing
train_i = int(len(main_data)*0.8)

train_data = main_data.loc[0:train_i]
test_data = main_data.loc[train_i + 1:]
#separate into X being the data of the site, Y being if it's fraudulent or not
#dont include the "url", use only the data of the site
#t: training ,ts ; testing
f = list(train_data.columns)
f.pop(0)
f.pop()
Xt = pd.DataFrame(train_data[f])

Yt = pd.DataFrame(train_data[["status"]])
Yt['status'] = [1 if x == 'phishing' else 0 for x in Yt['status']]

Xts = pd.DataFrame(test_data[f])

Yts = pd.DataFrame(test_data[["status"]])
Yts['status'] = [1 if y == 'phishing' else 0 for y in Yts['status']]

# now convert the data into numpy arrays so the AI can train
Xt = Xt.values
Yt = Yt.values

Xts = Xts.values
Yts = Yts.values


agent = IA(L=0.01,n=len(Xt))

agent.fit_function(Xt,Yt)

#test the agent and it´s performance
predictions = agent.predict(Xts)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

df = pd.DataFrame(zip(predictions,list(Yts.flatten())), columns=["pred","real"])
print(df)
df.to_csv('data_testing.csv',index=False)

acc = accuracy(predictions, Yts)
print("model accuraccy" , round(acc, 2), "%")
