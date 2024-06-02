
This is the code necessary to run experiments on GRelGT, submit to Information Systems, Under Review.


Please install relevant python packages before running the code. The settings are as follows:

```
dgl==0.4.2
lmdb==0.98
networkx==2.4
scikit-learn==0.22.1
torch==1.9.0
tqdm==4.43.0
```

To start training a model, run the following command. 
```
python train.py -d WN18RR_v1 -e grail_wn_v1

```
To test model, run the following commands.


```
python test_auc.py -d {dataset}_v{v}_ind -e {dataset}_v{v} --gpu {gpu} --num_neg_samples_per_link {num_neg_samples_per_link}'
```
The trained model and the logs are stored in experiments folder. Note that to ensure a fair comparison, we test all models on the same negative triplets. In order to do that in the current setup, we store the sampled negative triplets while evaluating GraIL and use these later to evaluate other baseline models.


When conducting training or testing during the experiment, you only need to convert to the corresponding functions of "train()" or "test()". The "num_neg_samples_per_link" in function "test()" means the number of negative samples during testing, which has two values (1 for the classification task and 50 for the ranking task) in our experiments. If you want to attempt other hyper-parameters, you can add the argument descriptions in function "train()" or change the corresponding values in "train.py".

Some codes are referenced by [GraIL](https://github.com/kkteru/grail) and [ConGLR](https://github.com/deepnolearning/ConGLR).

