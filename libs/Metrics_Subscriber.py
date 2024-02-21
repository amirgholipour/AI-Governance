## Define metrics

import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime, timezone, timedelta
from ibm_watson_machine_learning import APIClient
from ibm_wos_utils.explainability.utils.perturbations import Perturbations
from ibm_wos_utils.drift.drift_trainer import DriftTrainer
from libs.utils import save_as_json, read_from_json
from ibm_wos_utils.drift.drift_trainer import DriftTrainer
from json import dumps
from requests.auth import HTTPBasicAuth
import urllib3, requests, json, os,tarfile, time
from ibm_watson_openscale import *
from ibm_watson_openscale.supporting_classes.enums import *
from ibm_watson_openscale.supporting_classes import *
from ibm_watson_openscale.supporting_classes.payload_record import PayloadRecord
from datetime import datetime, timezone, timedelta
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import MonitorMeasurementRequest
# from dotenv import load_dotenv

class MetricsMonitor:
    def __init__(self, wos_client, wml_client, data_mart_id, subscription_id,config_json, class_label):
        # load_dotenv("assets/usecases/gcr/gcr_usecase.env")
        self.wos_client = wos_client
        self.wml_client = wml_client
        self.data_mart_id = data_mart_id
        self.subscription_id = subscription_id
        self.payload_data_set_id = None
        self.payload_scoring = None
        self.scoring_response = None
        self.config_json = config_json
        self.class_label = class_label
        self.ibm_cloud = False
        self.header = None
        self.quality_monitor_instance_id = None
        self.drift_monitor_instance_id = None
        self.fairness_monitor_instance_id = None
        self.custom_monitor_instance_id = None
        self.explanation_task_ids = None
        self.p_count=int(os.getenv("P_COUNT"))
        self.fairness_feature = os.getenv("FAIRNESS_FEATURE")
        self.features =  [{
            "feature": os.getenv("FAIRNESS_FEATURE"),
            "majority": [os.getenv("FAIRNESS_MAJORITY")],
            "minority": [os.getenv("FAIRNESS_MINORITY")],
            "correlated_attributes": []
        }]

        self.favourable_class = [os.getenv("FAIRNESS_FAVOURABLE_CLASS")]
        self.unfavourable_class = [os.getenv("FAIRNESS_UNFAVOURABLE_CLASS")]
        self.min_records = int(os.getenv("FAIRNESS_MIN_RECORDS"))
        self.quality_metric_threshold = float(os.getenv("QUALITY_METRIC_THRESHOLD"))
        self.min_feedback_data_size = int(os.getenv("MIN_FEEDBACK_DATA_SIZE"))

######################################################################################################################################      
####################################################    Generate Access token     ##################################################
######################################################################################################################################  
    def generate_access_token(self):
        headers = {"Content-Type": "application/json"}

        if self.ibm_cloud:
            grant_type = "urn:ibm:params:oauth:grant-type:apikey"
            api_key = os.environ.get('API_KEY', '')
            auth_data = {"grant_type": grant_type, "apikey": api_key}
            url = "https://iam.cloud.ibm.com/identity/token"
        else:
            grant_type = "password"
            username = os.environ.get('USERNAME', '')
            password = os.environ.get('PASSWORD', '')
            auth_data = {"grant_type": grant_type, "username": username, "password": password}
            url = os.environ.get("WOS_CLOUD_URL", "") + "/v1/preauth/validateAuth"

        response = requests.post(url, data=auth_data, headers=headers, verify=False)
        json_data = response.json()
        access_token = json_data.get('access_token', '')

        self.header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + access_token}
            
        

######################################################################################################################################      
####################################################      Explanation metrics      ##################################################
######################################################################################################################################  
    def explanation_metric(self):
    # score_perturbations_for_explanation(self):

        from ibm_wos_utils.explainability.utils.perturbations import Perturbations

        pertubation_count=int(self.p_count)
        perturbations=Perturbations(training_stats=self.config_json.get("explainability_configuration"), problem_type="binary", perturbations_count=pertubation_count)
        perturbs_df = perturbations.generate_perturbations()
        cols_to_remove = [self.class_label]
        
        def get_scoring_payload(no_of_records_to_score = 1):
            for col in cols_to_remove:
                if col in perturbs_df.columns:
                    del perturbs_df[col] 

            fields = perturbs_df.columns.tolist()
            values = perturbs_df[fields].values.tolist()

            payload_scoring = {"input_data": [{"fields": fields, "values": values[:no_of_records_to_score]}]}  
            return payload_scoring
        def sample_scoring(no_of_records_to_score = 1):
            records_list=[]
            payload_scoring = get_scoring_payload(no_of_records_to_score)
            # print(payload_scoring)
            scoring_response = self.wml_client.deployments.score(os.environ["DEPLOYMENT_ID"], payload_scoring)
            print('Single record scoring result:', '\n fields:', scoring_response['predictions'][0]['fields'], '\n values: ', scoring_response['predictions'][0]['values'][0])
            #print(json.dumps(scoring_response, indent=None))
            return payload_scoring, scoring_response
        self.payload_scoring, self.scoring_response = sample_scoring(no_of_records_to_score = 5000)



        
    # def create_explainability_archive(self):
        fields = self.scoring_response['predictions'][0]['fields']
        values = self.scoring_response['predictions'][0]['values']
        scored_data = pd.DataFrame(values, columns = fields)
        probabilities = [pro for pro in scored_data['probability']]
        predictions = [pre for pre in scored_data['prediction']]

        scored_perturbations = {'probabilities' : probabilities,
                                'predictions' : predictions}
        

        print(scored_perturbations)
        archive_data = {
                'lime_scored_perturbations.json': dumps(scored_perturbations),
                'training_statistics.json': dumps({"training_statistics": self.config_json.get("explainability_configuration")})
            }

        exp_filename = str(os.getenv("EXPLANABILITY_ARCHIVE_FILENAME"))+".tar.gz"

        with tarfile.open(exp_filename, mode="w:gz") as tf:
            for filename, filedata in archive_data.items():
                content = BytesIO(filedata.encode("utf8"))
                tarinfo = tarfile.TarInfo(filename)
                tarinfo.size = len(content.getvalue())
                tf.addfile(
                    tarinfo=tarinfo, fileobj=content)
        with open(exp_filename, mode='rb') as explainability_tar:
            self.wos_client.monitor_instances.upload_explainability_archive(subscription_id=self.subscription_id, archive=explainability_tar)

        print('Uploaded explainability configuration archive successfully.')

    # def enable_explainability_monitor(self):
        print("Creating explainability monitor instance...")
        target = Target(
            target_type=TargetTypes.SUBSCRIPTION,
            target_id=self.subscription_id
        )
        parameters={
            "enabled": True
        }
        response = self.wos_client.monitor_instances.create(monitor_definition_id=self.wos_client.monitor_definitions.MONITORS.EXPLAINABILITY.ID, 
                                                       data_mart_id=self.data_mart_id,
                                                       target=target,
                                                       parameters=parameters,
                                                       background_mode=False)
        print(response)
        
        self.payload_data_set_id = None
        self.payload_data_set_id = self.wos_client.data_sets.list(type=DataSetTypes.PAYLOAD_LOGGING, 
                                                        target_target_id=self.subscription_id, 
                                                        target_target_type=TargetTypes.SUBSCRIPTION).result.data_sets[0].metadata.id
        if self.payload_data_set_id is None:
            print("Payload data set not found. Please check subscription status.")
        else:
            print("Payload data set id: ", self.payload_data_set_id)
        

    # def trigger_local_explanation(self):
        payload_data = self.wos_client.data_sets.get_list_of_records(data_set_id=self.payload_data_set_id,output_type='pandas').result
        explanation_types = ["lime"]

        scoring_ids = payload_data.head(1)['scoring_id'].tolist()
        result = self.wos_client.monitor_instances.explanation_tasks(scoring_ids=scoring_ids, explanation_types=explanation_types, subscription_id=self.subscription_id).result

        self.explanation_task_ids=result.metadata.explanation_task_ids
        self.explanation_task_ids
        def finish_explanation_tasks(sample_size = 1):
            finished_explanations = []
            finished_explanation_task_ids = []

            # Check for the explanation task status for finished status. 
            # If it is in-progress state, then sleep for some time and check again. 
            # Perform the same for couple of times, so that all tasks get into finished state.
            for i in range(0, 5):
                # for each explanation
                print('iteration ' + str(i))

                #check status for all explanation tasks
                for explanation_task_id in self.explanation_task_ids:
                    if explanation_task_id not in finished_explanation_task_ids:
                        result = self.wos_client.monitor_instances.get_explanation_tasks(explanation_task_id=explanation_task_id, subscription_id=self.subscription_id).result
                        print(explanation_task_id + ' : ' + result.entity.status.state)
                        if (result.entity.status.state == 'finished' or result.entity.status.state == 'error') and explanation_task_id not in finished_explanation_task_ids:
                            finished_explanation_task_ids.append(explanation_task_id)
                            finished_explanations.append(result)


                # if there is altest one explanation task that is not yet completed, then sleep for sometime, 
                # and check for all those tasks, for which explanation is not yet completeed.

                if len(finished_explanation_task_ids) != sample_size:
                    print('sleeping for some time..')
                    time.sleep(10)
                else:
                    break

            return finished_explanations
        finished_explanations = finish_explanation_tasks(5)
        for result in finished_explanations:
            print(result)
            
            
######################################################################################################################################      
####################################################      Fairness metrics   #########################################################
######################################################################################################################################  
    def fairness_metric(self):
        ### Create Fairness Monitor Instance
        # Fairness configuration

        ### Create Fairness Monitor Instance
        target = Target(
            target_type=TargetTypes.SUBSCRIPTION,
            target_id=self.subscription_id

        )
        parameters = {
            "features": self.features,
            "favourable_class": self.favourable_class,
            "unfavourable_class": self.unfavourable_class,
            "min_records": self.min_records
        }

        thresholds = [{
            "metric_id": "fairness_value",
            "specific_values": [
                {
                    "applies_to": [{
                        "key": "feature",
                        "type": "tag",
                        "value": self.fairness_feature
                    }],
                    "value": float(os.getenv("FAIRNESS_THRESHOLD"))
                }
            ],
            "type": "lower_limit",
            "value": float(os.getenv("FAIRNESS_LIMIT"))
        }]

        fairness_monitor_details = self.wos_client.monitor_instances.create(
            data_mart_id=self.data_mart_id,
            background_mode=False,
            monitor_definition_id=self.wos_client.monitor_definitions.MONITORS.FAIRNESS.ID,
            target=target,
            parameters=parameters,
            thresholds=thresholds).result


        self.fairness_monitor_instance_id = fairness_monitor_details.metadata.id
        ### Get Fairness Monitor Instance

        self.wos_client.monitor_instances.show()
        ### Get run details
        ##### In case of production subscription, initial monitoring run is triggered internally. Checking its status
        runs = self.wos_client.monitor_instances.list_runs(self.fairness_monitor_instance_id, limit=1).result.to_dict()
        fairness_monitoring_run_id = runs["runs"][0]["metadata"]["id"]
        run_status = None
        while(run_status not in ["finished", "error"]):
            run_details = self.wos_client.monitor_instances.get_run_details(self.fairness_monitor_instance_id, fairness_monitoring_run_id).result.to_dict()
            run_status = run_details["entity"]["status"]["state"]
            print('run_status: ', run_status)
            if run_status in ["finished", "error"]:
                break
            time.sleep(10)
        ### Fairness run output
        self.wos_client.monitor_instances.get_run_details(self.fairness_monitor_instance_id, fairness_monitoring_run_id).result.to_dict()
        self.wos_client.monitor_instances.show_metrics(monitor_instance_id=self.fairness_monitor_instance_id)

        


        
        
######################################################################################################################################      
####################################################    Quality metrics      #########################################################
######################################################################################################################################     
    
    
    def quality_metric(self):
        # Quality monitoring and feedback logging
        ## Enable quality monitoring
        '''
        The code below waits ten seconds to allow the payload logging table to be set up before it begins enabling monitors. First, it turns on the quality (accuracy) monitor and sets an alert threshold of 70%. OpenScale will show an alert on the dashboard if the model accuracy measurement (area under the curve, in the case of a binary classifier) falls below this threshold.

        The second paramater supplied, min_records, specifies the minimum number of feedback records OpenScale needs before it calculates a new measurement. The quality monitor runs hourly, but the accuracy reading in the dashboard will not change until an additional 50 feedback records have been added, via the user interface, the Python client, or the supplied feedback endpoint.
        '''


        #time.sleep(10)
        target = Target(
                target_type=TargetTypes.SUBSCRIPTION,
                target_id=self.subscription_id
        )
        parameters = {
            "min_feedback_data_size": self.min_feedback_data_size # 100
        }
        thresholds = [
                        {
                            "metric_id": "area_under_roc",
                            "type": "lower_limit",
                            "value": self.quality_metric_threshold #.80
                        }
                    ]
        quality_monitor_details = self.wos_client.monitor_instances.create(
            data_mart_id=self.data_mart_id,
            background_mode=False,
            monitor_definition_id=self.wos_client.monitor_definitions.MONITORS.QUALITY.ID,
            target=target,
            parameters=parameters,
            thresholds=thresholds
        ).result

        self.quality_monitor_instance_id = quality_monitor_details.metadata.id
        # self.quality_monitor_instance_id

        ## Get feedback logging dataset ID
        feedback_dataset_id = None
        feedback_dataset = self.wos_client.data_sets.list(type=DataSetTypes.FEEDBACK, 
                                                        target_target_id=self.subscription_id, 
                                                        target_target_type=TargetTypes.SUBSCRIPTION).result
        feedback_dataset_id = feedback_dataset.data_sets[0].metadata.id
        if feedback_dataset_id is None:
            print("Feedback data set not found. Please check quality monitor status.")

        # feedback_payload_path = 'feedback_payload.json'

        # Read the data from the JSON file
        print(os.getcwd())
        feedback_payload = read_from_json(os.environ["FEEDBACK_DATA_PATH"])
        if self.ibm_cloud ==True:

            DATASETS_STORE_RECORDS_URL = "https://api.aiopenscale.cloud.ibm.com/openscale/{0}/v2/data_sets/{1}/records".format(self.data_mart_id, feedback_dataset_id)
            for x in range(10):
                response = requests.post(DATASETS_STORE_RECORDS_URL, json=feedback_payload, headers=self.header, verify=False)
                json_data = response.json()
                print(json_data)

            ### Wait for sometime, and make sure the records have reached to data sets related table.
            time.sleep(10)
            DATASETS_STORE_RECORDS_URL = "https://api.aiopenscale.cloud.ibm.com/openscale/{0}/v2/data_sets/{1}/records?limit={2}&include_total_count={3}".format(self.data_mart_id, feedback_dataset_id, 1, "true")
            response = requests.get(DATASETS_STORE_RECORDS_URL, headers=self.header, verify=False)
            json_data = response.json()
            print(json_data['total_count'])
        else:
            
            self.wos_client.data_sets.store_records(feedback_dataset_id, request_body=[feedback_payload], background_mode=False)
            self.wos_client.data_sets.get_records_count(data_set_id=feedback_dataset_id)
            print(self.wos_client.data_sets.get_records_count(data_set_id=feedback_dataset_id))
        
        ## Run Quality Monitor
        run_details = self.wos_client.monitor_instances.run(monitor_instance_id=self.quality_monitor_instance_id, background_mode=False).result
        self.wos_client.monitor_instances.show_metrics(monitor_instance_id=self.quality_monitor_instance_id)

        

        
    
    
    
######################################################################################################################################      
####################################################      Drift metrics      #########################################################
######################################################################################################################################   
    
    def drift_metric(self,feature_columns,cat_features):
        # Drift Configuration
        ## Create the drift detection model archive

        # feature_columns=["CheckingStatus","LoanDuration","CreditHistory","LoanPurpose","LoanAmount","ExistingSavings","EmploymentDuration","InstallmentPercent","Sex","OthersOnLoan","CurrentResidenceDuration","OwnsProperty","Age","InstallmentPlans","Housing","ExistingCreditsCount","Job","Dependents","Telephone","ForeignWorker"]
        len(feature_columns)

        def score(training_data_frame):

            #The data type of the label column and prediction column should be same .
            #User needs to make sure that label column and prediction column array should have the same unique class labels
            prediction_column_name = "prediction"
            probability_column_name = "probability"

            # feature_columns = list(feature_columns.columns)
            training_data_rows = training_data_frame[feature_columns].values.tolist()

            payload_scoring = {
              self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{
                   "fields": feature_columns,
                   "values": [x for x in training_data_rows]
              }]
            }

            # print(payload_scoring)

            score = self.wml_client.deployments.score(os.environ["DEPLOYMENT_ID"], payload_scoring)
            score_predictions = score.get('predictions')[0]

            prob_col_index = list(score_predictions.get('fields')).index(probability_column_name)
            predict_col_index = list(score_predictions.get('fields')).index(prediction_column_name)

            if prob_col_index < 0 or predict_col_index < 0:
                raise Exception("Missing prediction/probability column in the scoring response")

            
            probability_array = np.array([value[prob_col_index] for value in score_predictions.get('values')])
            prediction_vector = np.array([value[predict_col_index] for value in score_predictions.get('values')])

            return probability_array, prediction_vector


        
        training_data_frame=pd.read_csv(str(os.getenv("TRAINING_DATA_FILE"))+".csv")
        # training_data_frame.head()



        probability_array, prediction_vector = score(training_data_frame)
        # prediction_vector


        

        drift_detection_input = {
            "feature_columns": feature_columns,
            "categorical_columns":cat_features,
            "label_column": self.class_label,
            "problem_type": "binary"
        }

        drift_trainer = DriftTrainer(training_data_frame, drift_detection_input)
        drift_trainer.generate_drift_detection_model(score, batch_size=training_data_frame.shape[0], check_for_ddm_quality=False)
        drift_trainer.learn_constraints(
            two_column_learner_limit=200, categorical_unique_threshold=0.8, user_overrides=[])
        drift_trainer.create_archive()

        ### Enable the drift monitor
        self.wos_client.monitor_instances.upload_drift_model(
            model_path=str(os.getenv("DRIFT_MODEL"))+".tar.gz",
            data_mart_id=self.data_mart_id,
            subscription_id=self.subscription_id
        ).result

        import time

        target = Target(
            target_type=TargetTypes.SUBSCRIPTION,
            target_id=self.subscription_id
        )

        parameters = {
            "min_samples": int(os.getenv("DRIFT_MIN_SAMPLES")),
            "drift_threshold": float(os.getenv("DRIFT_THRESHOLD")),
            "train_drift_model": False
        }

        drift_monitor_details = self.wos_client.monitor_instances.create(
            data_mart_id=self.data_mart_id,
            monitor_definition_id=self.wos_client.monitor_definitions.MONITORS.DRIFT.ID,
            target=target,
            parameters=parameters
        ).result

        self.drift_monitor_instance_id = drift_monitor_details.metadata.id
        print(drift_monitor_details)

        ## Check monitor instance status
        drift_status = None

        while drift_status not in ("active", "error"):
            monitor_instance_details = self.wos_client.monitor_instances.get(monitor_instance_id=self.drift_monitor_instance_id).result
            drift_status = monitor_instance_details.entity.status.state
            if drift_status not in ("active", "error"):
                print(datetime.utcnow().strftime('%H:%M:%S'), drift_status)
                time.sleep(30)

        print(datetime.utcnow().strftime('%H:%M:%S'), drift_status)
        ## Run an on-demand evaluation
        # Check Drift monitor instance details

        monitor_instance_details = self.wos_client.monitor_instances.get(monitor_instance_id=self.drift_monitor_instance_id).result
        print(monitor_instance_details)
        # Trigger on-demand run

        monitoring_run_details = self.wos_client.monitor_instances.run(monitor_instance_id=self.drift_monitor_instance_id).result
        monitoring_run_id=monitoring_run_details.metadata.id

        print(monitoring_run_details)

        monitoring_run_details = self.wos_client.monitor_instances.get_run_details(monitor_instance_id=self.drift_monitor_instance_id, monitoring_run_id=monitoring_run_id).result
        monitoring_run_details.entity.status.state

        # Check run status

        drift_run_status = None
        while drift_run_status not in ("finished", "error"):
            monitoring_run_details = self.wos_client.monitor_instances.get_run_details(monitor_instance_id=self.drift_monitor_instance_id, monitoring_run_id=monitoring_run_id).result
            drift_run_status = monitoring_run_details.entity.status.state
            if drift_run_status not in ("finished", "error"):
                print(datetime.utcnow().strftime("%H:%M:%S"), drift_run_status)
                time.sleep(30)

        print(datetime.utcnow().strftime("%H:%M:%S"), drift_run_status)

        ## Display drift metrics
        self.wos_client.monitor_instances.show_metrics(monitor_instance_id=self.drift_monitor_instance_id)
######################################################################################################################################      
####################################################      Custom metrics      ########################################################
######################################################################################################################################
    def custom_metric(self):
        # Custom monitors and metrics <a name="custom"></a>
        ## Register custom monitor
        def get_definition(monitor_name):
            monitor_definitions = self.wos_client.monitor_definitions.list().result.monitor_definitions

            for definition in monitor_definitions:
                if monitor_name == definition.entity.name:
                    return definition

            return None

        monitor_name = os.getenv("CUSTOM_MONITOR_NAME")
        metrics = [MonitorMetricRequest(name='sensitivity',
                                        thresholds=[MetricThreshold(type=MetricThresholdTypes.LOWER_LIMIT, default=0.8)]),
                  MonitorMetricRequest(name='specificity',
                                        thresholds=[MetricThreshold(type=MetricThresholdTypes.LOWER_LIMIT, default=0.75)])]
        tags = [MonitorTagRequest(name='region', description='customer geographical region')]

        existing_definition = get_definition(monitor_name)

        if existing_definition is None:
            custom_monitor_details = self.wos_client.monitor_definitions.add(name=monitor_name, metrics=metrics, tags=tags, background_mode=False).result
        else:
            custom_monitor_details = existing_definition

        self.wos_client.monitor_definitions.show()
        ### Get monitors uids and details
        custom_monitor_id = custom_monitor_details.metadata.id

        print(custom_monitor_id)
        custom_monitor_details = self.wos_client.monitor_definitions.get(monitor_definition_id=custom_monitor_id).result
        print('Monitor definition details:', custom_monitor_details)
        ## Enable custom monitor for subscription
        target = Target(
                target_type=TargetTypes.SUBSCRIPTION,
                target_id=self.subscription_id
            )

        thresholds = [MetricThresholdOverride(metric_id='sensitivity', type = MetricThresholdTypes.LOWER_LIMIT, value=0.9)]
        parameters = {
            "min_feedback_data_size": os.getenv("CUSTOM_FEEDBACK_DATA_SIZE")
        }
        custom_monitor_instance_details = self.wos_client.monitor_instances.create(
                    data_mart_id=self.data_mart_id,
                    background_mode=False,
                    monitor_definition_id=custom_monitor_id,
                    parameters=parameters,
            thresholds=thresholds,
                    target=target
        ).result

        ### Get monitor instance id and configuration details
        self.custom_monitor_instance_id = custom_monitor_instance_details.metadata.id
        custom_monitor_instance_details = self.wos_client.monitor_instances.get(self.custom_monitor_instance_id).result
        print(custom_monitor_instance_details)
        ## Storing custom metrics

        custom_monitoring_run_id = os.getenv("CUSTOM_MONITOR_ID")
        specificity=float(os.getenv("CUSTOM_SPECIFICITY"))
        sensitivity=float(os.getenv("CUSTOM_SENSITIVITY"))
        region=os.getenv("CUSTOM_REGION")

        measurement_request = [MonitorMeasurementRequest(timestamp=datetime.now(timezone.utc), 
                                                         metrics=[{"specificity": specificity, "sensitivity": sensitivity, "region": region}], run_id=custom_monitoring_run_id)]

        # measurement_request = [MonitorMeasurementRequest(timestamp=datetime.now(timezone.utc), 
        #                                                  metrics=[{"specificity": 0.76, "sensitivity": 0.75, "region": "us-south"}], run_id=custom_monitoring_run_id)]
        print(measurement_request[0])

        published_measurement_response = self.wos_client.monitor_instances.measurements.add(
            monitor_instance_id=self.custom_monitor_instance_id,
            monitor_measurement_request=measurement_request).result
        published_measurement_id = published_measurement_response[0]["measurement_id"]
        print(published_measurement_response)
        ### List and get custom metrics
        time.sleep(5)
        published_measurement = self.wos_client.monitor_instances.measurements.get(monitor_instance_id=self.custom_monitor_instance_id, measurement_id=published_measurement_id).result
        print(published_measurement)




