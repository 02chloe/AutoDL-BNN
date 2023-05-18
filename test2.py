import openml
# from pprint import pprint
from autoPyTorch import AutoNetClassification, AutoNetMultilabel
from sklearn.metrics import accuracy_score
import time

# get OpenML task by its ID
task_id = 146212
task = openml.tasks.get_task(task_id=146212)
X, y = task.get_X_and_y()
ind_train, ind_test = task.get_train_test_split_indices()


# run Auto-PyTorch
autoPyTorch = AutoNetClassification("medium_cs",  # config preset
                                    log_level='info',
                                    max_runtime=600,
                                    min_budget=30,
                                    max_budget=180)
start = time.time()
autoPyTorch.fit(X[ind_train], y[ind_train], validation_split=0.3)
end = time.time() 
fit_time = end-start
# predict
y_pred = autoPyTorch.predict(X[ind_test])


score = accuracy_score(y[ind_test], y_pred)

print("Accuracy score", score)
print('Run_time:',fit_time)
print('task-id:',task_id)

# print network configuration
#pprint(autoPyTorch.fit_result["optimized_hyperparameter_config"])
with open('/user/work/aj20377/workshop/apm-146212-score.txt','a') as file:    
    file.write(str(score))
    file.write('\n')

with open('/user/work/aj20377/workshop/apm-146212-time.txt','a') as file:   
    file.write(str(fit_time))
    file.write('\n')


