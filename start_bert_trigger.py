import os
import mlflow


if __name__ == "__main__":
    experiment_id = mlflow.create_experiment("BertML-experiment")

    command_str = "ssh zhoufang@10.101.7.1 '/home/common/software/anaconda3/bin/python3 -u /home/zhoufang/mpt-wx/yu_bert_train_trigger.py'"
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("iteration", "100")
        mlflow.log_param("batch size", "8")
        mlflow.log_metric("learning rate", "0.0000009173")
        mlflow.log_metric("lm loss", "0.000008050369")
        mlflow.log_metric("sop loss", "0.0000007277749")
        os.system(command_str)
        