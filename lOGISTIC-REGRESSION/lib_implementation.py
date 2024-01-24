import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#load the data:#read the data using dataframes
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
Yts = Yts.values.flatten()

model = LogisticRegression(solver='liblinear', random_state=0).fit(Xt, Yt)

test_results = model.predict(Xts)

accuracy = np.sum(test_results==Yts)/Yts.size
print('Accuracy of the model:', accuracy)