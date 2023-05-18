import torch
torch.set_default_dtype(torch.float64)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Optim

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import openml
openml.config.apikey = 'd2ddc532ae7c1a7e3f5c36766b2237bb'

id = 14965
task = openml.tasks.get_task(task_id=14965)
X, y = task.get_X_and_y()
X[np.isnan(X)]=0
X = X.astype(np.float64)
xx = np.insert(X,X.shape[1],y,axis=1)
DATA = pd.DataFrame(xx)

# DATA = DATA.sample(frac=1).reset_index(drop=True)
# #DATA.info()

class DataSet(torch.utils.data.dataset.Dataset):
    def __init__(self, df):

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        features = self.df.iloc[idx, :-1].values.astype(np.float64)
        outcomes = np.array(self.df.iloc[idx, -1]).astype(np.int64)
        example = {"features": features, "outcomes": outcomes}
        return example
    
test_num = int(len(DATA)*0.2)
train_dataset = DataSet(DATA[:-test_num])
test_dataset = DataSet(DATA[-test_num:])

outcomes_num = len(np.unique(np.array(DATA.iloc[:,-1])))
feature_num = DATA.shape[1]-1
# print(feature_num,outcomes_num)
# train_dataset[0]

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=50,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=50,
                                          shuffle=True, num_workers=2)

class DenseModel(nn.Module):
    def __init__(self, feature_num, outcomes_num):

        super(DenseModel, self).__init__()

        self.fc1 = nn.Linear(feature_num, outcomes_num)
        self.fc2 = nn.Linear(outcomes_num, outcomes_num)
        self.fc3 = nn.Linear(outcomes_num, outcomes_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

dense = DenseModel(feature_num, outcomes_num)
## print(dense)

learning_rate=0.001
criterion = nn.CrossEntropyLoss()
optimizer = Optim.Adam(dense.parameters(), lr=learning_rate)


def model(data):
    fc1w_prior = dist.Normal(loc=torch.zeros_like(dense.fc1.weight), scale=torch.ones_like(dense.fc1.weight))
    fc1b_prior = dist.Normal(loc=torch.zeros_like(dense.fc1.bias), scale=torch.ones_like(dense.fc1.bias))
    
    fc2w_prior = dist.Normal(loc=torch.zeros_like(dense.fc2.weight), scale=torch.ones_like(dense.fc2.weight))
    fc2b_prior = dist.Normal(loc=torch.zeros_like(dense.fc2.bias), scale=torch.ones_like(dense.fc2.bias))
    
    fc3w_prior = dist.Normal(loc=torch.zeros_like(dense.fc3.weight), scale=torch.ones_like(dense.fc3.weight))
    fc3b_prior = dist.Normal(loc=torch.zeros_like(dense.fc3.bias), scale=torch.ones_like(dense.fc3.bias))
    
    priors = {"fc1w": fc1w_prior, "fc1b": fc1b_prior,  "fc3w": fc3w_prior, "fc3b": fc3b_prior}

    lifted_module = pyro.random_module("module", dense, priors)
    lifted_reg_model = lifted_module()
    
    probs = torch.nn.functional.log_softmax(lifted_reg_model(data["features"]),dim=1)
    
    pyro.sample("obs", dist.Categorical(logits=probs), obs=data["outcomes"])
    

def guide(data):
    
    # FC1 weights
    fc1w_mu = torch.randn_like(dense.fc1.weight)
    fc1w_sigma = torch.randn_like(dense.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = F.softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_approx_post = dist.Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    # FC1 bias
    fc1b_mu = torch.randn_like(dense.fc1.bias)
    fc1b_sigma = torch.randn_like(dense.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = F.softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_approx_post = dist.Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
    
    # FC2 weights
    fc2w_mu = torch.randn_like(dense.fc2.weight)
    fc2w_sigma = torch.randn_like(dense.fc2.weight)
    fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    fc2w_sigma_param = F.softplus(pyro.param("fc2w_sigma", fc2w_sigma))
    fc2w_approx_post = dist.Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
    # FC2 bias
    fc2b_mu = torch.randn_like(dense.fc2.bias)
    fc2b_sigma = torch.randn_like(dense.fc2.bias)
    fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    fc2b_sigma_param = F.softplus(pyro.param("fc2b_sigma", fc2b_sigma))
    fc2b_approx_post = dist.Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)
    
    # FC3 weights
    fc3w_mu = torch.randn_like(dense.fc3.weight)
    fc3w_sigma = torch.randn_like(dense.fc3.weight)
    fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
    fc3w_sigma_param = F.softplus(pyro.param("fc3w_sigma", fc3w_sigma))
    fc3w_approx_post = dist.Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param).independent(1)
    # FC3 bias
    fc3b_mu = torch.randn_like(dense.fc3.bias)
    fc3b_sigma = torch.randn_like(dense.fc3.bias)
    fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
    fc3b_sigma_param = F.softplus(pyro.param("fc3b_sigma", fc3b_sigma))
    fc3b_approx_post = dist.Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param)
    priors = {"fc1w": fc1w_approx_post, "fc1b": fc1b_approx_post, "fc2.weight":fc2w_approx_post, "fc2.bias":fc2b_approx_post,  "fc3w": fc3w_approx_post, "fc3b": fc3b_approx_post}
    # approx_posterior = {"fc1w": fc1w_approx_post, "fc1b": fc1b_approx_post,  "fc3w": fc3w_approx_post, "fc3b": fc3b_approx_post}

    lifted_module = pyro.random_module("module", dense, priors)
    # lifted_module = pyro.random_module("module", dense, approx_posterior)
    
    return lifted_module()

optim = pyro.optim.Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

num_iterations = 100
loss = 0

fit_s = time.time()
for j in range(num_iterations):
    loss = 0
    for batch_id, data in enumerate(trainloader):
        # calculate the loss and take a gradient step
        loss += svi.step(data)
    normalizer_train = len(trainloader.dataset)
    total_epoch_loss_train = loss / normalizer_train    
    print("Epoch ", j, " Loss ", total_epoch_loss_train)
fit_e = time.time()
fit_t = fit_e - fit_s

num_samples = 3

pre_s = time.time()
def predict(Data):
    sampled_models = [guide(None) for _ in range(num_samples)]
    predictions = [m(Data).data for m in sampled_models]
    mean = torch.mean(torch.stack(predictions), 0)    
    return np.argmax(mean.numpy(), axis=1)
pre_e = time.time()
pre_t = pre_e - pre_s

correct = 0
total = 0
re = np.empty(0)
ori = np.empty(0)

for j, data in enumerate(testloader): 
    features = data["features"]
    labels = data["outcomes"]
    predicted = predict(features)
    total += labels.size(0)
    ori = np.hstack((ori, np.array(labels)))
    re = np.hstack((re, predicted))
    correct += (np.array(predicted) == np.array(labels)).sum().item()
print("accuracy: {:.2f}".format(100 * correct / total))
score = accuracy_score(ori, re, normalize=True)
print('task-id:',id)
print('accuracy score:',score)
# print('fit_time',fit_t)
# print('predict time',pre_t)
# print('feature_num:',feature_num)
# print('class_num:',outcomes_num)

# print('data_num:',len(DATA))
# print('train_num:',len(train_dataset),'test_num:',len(test_dataset))

# print('num_iterations',num_iterations)

# score = 'bnn-53'
# fit_t = 'bnn-53'
with open('/user/work/aj20377/workshop/bnn2-14965-score.txt','a') as file: 
    file.write(str(score))
    file.write('\n')

with open('/user/work/aj20377/workshop/bnn2-14965-time.txt','a') as file:    
    file.write(str(fit_t))
    file.write('\n')


