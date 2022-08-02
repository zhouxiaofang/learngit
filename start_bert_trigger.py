import os
import mlflow

if __name__ == "__main__":
    command_str = "ssh zhoufang@10.101.7.1 '/home/common/software/anaconda3/bin/python3 -u /home/zhoufang/yu_bert_train_trigger.py'"
    with mlflow.start_run():
        os.system(command_str)
        mlflow.log_param("alpha", "0.5")