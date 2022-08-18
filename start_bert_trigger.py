import os
import mlflow
from decimal import Decimal


if __name__ == "__main__":
    experiment_id = mlflow.create_experiment("BertML-experiment5")
    
    rate = float(Decimal(0.9173))

    command_str = "ssh zhoufang@10.101.7.1 '/home/common/software/anaconda3/bin/python3 -u /home/zhoufang/mpt-wx/yu_bert_train_trigger.py'"
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("iteration", "100")
        mlflow.log_param("batch size", "8")
        mlflow.log_metric("learning rate", rate)
        mlflow.log_metric("lm loss", rate)
        mlflow.log_metric("sop loss", rate)
        os.system(command_str)
        