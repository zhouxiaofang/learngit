import os
import mlflow


if __name__ == "__main__":
    experiment_id = mlflow.create_experiment("bert-experiment")

    command_str = "ssh zhoufang@10.101.7.1 '/home/common/software/anaconda3/bin/python3 -u /home/zhoufang/yu_bert_train_trigger.py'"
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("alpha", "5.0")
        mlflow.log_param("l1_ratio", "0.1")
        mlflow.log_metric("rmse", "0.859")
        mlflow.log_metric("r2", "0.047")
        mlflow.log_metric("mae", "0.648")
        os.system(command_str)
        