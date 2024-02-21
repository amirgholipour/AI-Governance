#Get training data statistics

import pandas as pd
from ibm_watson_openscale.utils.training_stats import TrainingStats
import os
import requests
import subprocess
import json
class ReferenceDataAnalyser:
    def __init__(self):
        self.feature_columns = json.loads(os.getenv("FEATURE_COLUMNS"))
        self.cat_features = json.loads(os.getenv("CAT_FEATURES"))
        self.class_label = os.getenv("CLASS_LABEL")
        
        self.data_df = None
        self.config_json = None
        self.dataset_path = os.getenv("DATASET_PATH")
        if os.getenv("PROBLEM_TYPE") == "BINARY_CLASSIFICATION":
            self.problem_type = "binary"

    def collect_reference_data(self):
        
        
        

        # # # Define the command and arguments
        # # command = "rm"
        # # args = ["-lh", "german_credit_feed.json"]

        # # Run the command
        # # result = subprocess.run([command] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # # # Print the output
        # # if result.returncode == 0:
        # #     print(result.stdout)
        # # else:
        # #     print(f"Error: {result.stderr}")
        
        # # File to remove
        # file_to_remove = 'german_credit_data_biased_training.csv'

        # # Remove the file if it exists
        # if os.path.exists(file_to_remove):
        #     subprocess.run(['rm', file_to_remove])

        # # URL to download the file from
        # url = "https://raw.githubusercontent.com/pmservice/ai-openscale-tutorials/master/assets/historical_data/german_credit_risk/wml/german_credit_data_biased_training.csv"

        # # Download the file using wget
        # subprocess.run(['wget', url])
            
    
        # # !rm german_credit_data_biased_training.csv
        # # !wget https://raw.githubusercontent.com/pmservice/ai-openscale-tutorials/master/assets/historical_data/german_credit_risk/wml/german_credit_data_biased_training.csv
    
        # self.data_df = pd.read_csv(url)
        self.data_df = pd.read_csv(self.dataset_path)
        return self.data_df.head()

    def generate_reference_config(self):
        if self.data_df is None:
            raise ValueError("Training data not loaded. Call collect_training_data first.")
        
        input_parameters = {
            "label_column": self.class_label,
            "feature_columns": self.feature_columns,
            "categorical_columns": self.cat_features,
            "fairness_inputs": None, 
            "prediction_column": self.class_label,
            "probability_column": "probability",
            "problem_type" : self.problem_type
        }
        training_stats = TrainingStats(self.data_df, input_parameters, explain=True, fairness=False, drop_na=True)
        self.config_json = training_stats.get_training_statistics()
        self.config_json["notebook_version"] = 6.0
        return self.config_json
