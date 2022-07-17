import torch

# average f1 score 
def f1_score(conf_mat):
    prediction_sum = torch.sum(conf_mat, dim=0)
    target_sum = torch.sum(conf_mat, dim=1)

    f_scores = []

    for i in range(5):
        f = 2 * conf_mat[i,i]
        f /= (prediction_sum[i] + target_sum[i])
        f_scores.append(f)
    
    f1 = sum(f_scores) / 5
    
    return f1.item()

# confusion matrix
def confusion_mat(prediction, target):
    conf_mat = torch.zeros(5, 5)
    
    prediction = torch.argmax(prediction, dim=1)
    for i in range(prediction.size(0)):
        conf_mat[prediction[i],target[i]] += 1
    return conf_mat
        