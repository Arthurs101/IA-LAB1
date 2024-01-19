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


agent = IA(L=0.0015,n=len(Xt))

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



correlation["result"] = predictions.tolist()


import matplotlib.pyplot as plt



phishing_color = "red"  # Color for "1" (phishing)
non_phishing_color = "green"  # Color for "0" (no phishing)

plt.figure(figsize=(10, 6))

for i, row in correlation.iterrows():
    x = row["length_url"]
    y = row["web_traffic"]
    color = phishing_color if row["result"] == 1 else non_phishing_color

    plt.scatter(x, y, c=color, s=40)

plt.xlabel("Length URL")
plt.ylabel("Web Traffic")
plt.title("Correlation of Length URL and Web Traffic with Phishing Classification")

plt.grid(True)

# Optional legend
legend_labels = ["Phishing", "No Phishing"]
legend_handles = [
    plt.Circle((0, 0), 1, color=phishing_color),
    plt.Circle((0, 0), 1, color=non_phishing_color),
]
plt.legend(legend_handles, legend_labels, title="Classification")

plt.show()
