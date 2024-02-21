import os
import time
import uuid
from ibm_watson_openscale import *
from ibm_watson_openscale.supporting_classes.enums import *
from ibm_watson_openscale.supporting_classes import *
from ibm_watson_openscale.supporting_classes.payload_record import PayloadRecord
from libs.utils import save_as_json, read_from_json

class ModelSubscriber:
    def __init__(self, wos_client):
        self.wos_client = wos_client
        self.data_mart_id = None
        self.service_provider_id = None
        self.subscription_id = None
        self.payload_data_set_id = None
        
        self.file_sreq_path = os.getenv("SCORING_REQUEST_FILE")
        self.file_sres_path = os.getenv("SCORING_RESPONSE_FILE")
        

        if os.getenv("PROBLEM_TYPE") == "BINARY_CLASSIFICATION":
            self.problem_type = ProblemType.BINARY_CLASSIFICATION
            
        if os.getenv("INPUT_DATA_TYPE")== "STRUCTURED":
            self.input_data_type=InputDataType.STRUCTURED
            
        if os.getenv("DEPLOYMENT_TYPES")== "ONLINE":    
            self.deployment_type=DeploymentTypes.ONLINE
        else:
            self.deployment_type=DeploymentTypes.BATCH

    def setup_datamart(self):
        print(self.wos_client.data_marts.show())
        data_marts = self.wos_client.data_marts.list().result.data_marts
        self.data_mart_id = data_marts[0].metadata.id
        print('Using existing datamart {}'.format(self.data_mart_id))

    def add_service_provider(self):
        # Remove existing service provider
        service_providers = self.wos_client.service_providers.list().result.service_providers
        for service_provider in service_providers:
            if service_provider.entity.name == os.getenv("SERVICE_PROVIDER_NAME"):
                self.wos_client.service_providers.delete(service_provider.metadata.id)
                print("Deleted existing service_provider with ID: {}".format(service_provider.metadata.id))
        
        # Add new service provider
        added_service_provider_result = self.wos_client.service_providers.add(
            name=os.getenv("SERVICE_PROVIDER_NAME"),
            description=os.getenv("SERVICE_PROVIDER_DESCRIPTION"),
            service_type=ServiceTypes.CUSTOM_MACHINE_LEARNING,
            operational_space_id="production",
            credentials={},
            background_mode=False
        ).result
        self.service_provider_id = added_service_provider_result.metadata.id
        print("Added new service provider with ID: {}".format(self.service_provider_id))

    def subscribe_model(self, config_json):
        # Remove existing subscription
        subscriptions = self.wos_client.subscriptions.list().result.subscriptions
        for subscription in subscriptions:
            if subscription.entity.asset.name == os.getenv("SUBSCRIPTION_NAME"):
                self.wos_client.subscriptions.delete(subscription.metadata.id)
                print('Deleted existing subscription with ID:', subscription.metadata.id)
        
        # Subscribe model
        asset_id = str(uuid.uuid4())
        asset_name = os.getenv("SUBSCRIPTION_NAME")
        asset_deployment_id = str(uuid.uuid4())
        
            
        subscription_details = self.wos_client.subscriptions.add(
            self.data_mart_id,
            self.service_provider_id,
            asset=Asset(
                asset_id=asset_id,
                name=asset_name,
                url='',
                asset_type=AssetTypes.MODEL,
                input_data_type=self.input_data_type,
                problem_type=self.problem_type
            ),
            deployment=AssetDeploymentRequest(
                deployment_id=asset_deployment_id,
                name=asset_name,
                deployment_type=self.deployment_type
            ),
            training_data_stats=config_json,
            prediction_field="prediction",
            predicted_target_field="prediction",
            probability_fields=['probability'],
            background_mode=False,
            deployment_name=asset_name
        ).result
        self.subscription_id = subscription_details.metadata.id
        print("Subscribed model with ID: {}".format(self.subscription_id))

    def push_payload_record(self):
        
        
        scoring_request = read_from_json(self.file_sreq_path)
        scoring_response = read_from_json(self.file_sres_path)
        time.sleep(5)
        self.payload_data_set_id = None
        self.payload_data_set_id = self.wos_client.data_sets.list(type=DataSetTypes.PAYLOAD_LOGGING, 
                                                        target_target_id=self.subscription_id, 
                                                        target_target_type=TargetTypes.SUBSCRIPTION).result.data_sets[0].metadata.id
        if self.payload_data_set_id is None:
            print("Payload data set not found. Please check subscription status.")
        else:
            print("Payload data set id: ", self.payload_data_set_id)
        
        records_list = [PayloadRecord(request=scoring_request, response=scoring_response) for _ in range(10)]
        self.wos_client.data_sets.store_records(data_set_id=self.payload_data_set_id, request_body=records_list)

        time.sleep(15)
        pl_records_count = self.wos_client.data_sets.get_records_count(self.payload_data_set_id)
        print("Number of records in the payload logging table: {}".format(pl_records_count))
        if pl_records_count == 0:
            raise Exception("Payload logging did not happen!")

    def verify_subscription(self):
        return self.wos_client.subscriptions.get(self.subscription_id).result.to_dict()

