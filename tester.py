import numpy as np
import pandas as pd
from logical_regrex import logical_Rgrex as IA
import matplotlib as plt
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


agent = IA(L=0.001,n=len(Xt))

agent.fit_function(Xt,Yt)

#test the agent and it´s performance
predictions = agent.predict(Xts)

df = pd.DataFrame(zip(predictions,list(Yts.flatten())), columns=["pred","real"])

df.to_csv('data_testing.csv',index=False)
tp = np.sum((predictions == 1) & (Yts == 1))
tn = np.sum((predictions == 0) & (Yts == 0))
fp = np.sum((predictions == 1) & (Yts == 0))
fn = np.sum((predictions == 0) & (Yts == 1))

acc = (tp + tn) / (tp + tn + fn + fp)* 100 

print("model accuraccy" , round(acc, 2), "%")

#graphics analysis
correlation = pd.DataFrame(test_data[["length_url","web_traffic"]])
correlation = correlation.reset_index(drop=True)

def get_result_label(prediction, real_value):
    if prediction == 1 and real_value == 1:
        return "tp"
    elif prediction == 0 and real_value == 0:
        return "tn"
    elif prediction == 1 and real_value == 0:
        return "fp"
    else:
        return "fn"

correlation["result"] = [get_result_label(p, y) for p, y in zip(predictions, Yts)]


import matplotlib.pyplot as plt



colors = {"tp": "green", "tn": "blue", "fp": "red", "fn": "orange"}  # Color mapping for labels

plt.figure(figsize=(10, 6))  

for i, row in correlation.iterrows():
    x = row["length_url"]
    y = row["web_traffic"]
    color = colors[row["result"]]

    plt.scatter(x, y, c=color, s=40) 

plt.xlabel("Length URL")
plt.ylabel("Web Traffic")
plt.title("Correlación de clasificacion basado en largo del url y su tráfico como fraudulento")

plt.grid(True) 
legend_labels = list(list(colors.keys()))
legend_handles = [plt.Circle((0, 0), 1, color=colors[label]) for label in legend_labels]

plt.legend(legend_handles, legend_labels, title="Classification Result")
plt.show()
