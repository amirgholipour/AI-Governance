## Define metrics


from json import dumps
import random
import  json, os, time,uuid
from ibm_watson_openscale import *
from ibm_watson_openscale.supporting_classes.enums import *
from ibm_watson_openscale.supporting_classes import *
from ibm_watson_openscale.supporting_classes.payload_record import PayloadRecord
# from dotenv import load_dotenv


class PayloadLogging:
    def __init__(self, wos_client, wml_client, subscription_id):
        # load_dotenv()
        self.wos_client = wos_client
        self.wml_client = wml_client
        self.subscription_id = subscription_id
        self.payload_data_set_id = None
        self.payload_log_file=os.getenv("PAYLOAD_LOG_FILE")
        

    def log_payload(self):

        with open(self.payload_log_file, 'r') as scoring_file:
            scoring_data = json.load(scoring_file)

        fields = scoring_data['fields']
        values = []
        for _ in range(1000):
            values.append(random.choice(scoring_data['values']))
        payload_scoring = {"input_data": [{"fields": fields, "values": values}]}

        scoring_response = self.wml_client.deployments.score(os.environ["DEPLOYMENT_ID"], payload_scoring)

        
        time.sleep(5)
        self.payload_data_set_id = None
        self.payload_data_set_id = self.wos_client.data_sets.list(type=DataSetTypes.PAYLOAD_LOGGING, 
                                                        target_target_id=self.subscription_id, 
                                                        target_target_type=TargetTypes.SUBSCRIPTION).result.data_sets[0].metadata.id
        if self.payload_data_set_id is None:
            print("Payload data set not found. Please check subscription status.")
        else:
            print("Payload data set id: ", self.payload_data_set_id)
        time.sleep(5)
        pl_records_count = self.wos_client.data_sets.get_records_count(self.payload_data_set_id)

        print("Performing explicit payload logging.")
        self.wos_client.data_sets.store_records(data_set_id=self.payload_data_set_id, request_body=[PayloadRecord(
                    scoring_id=str(uuid.uuid4()),
                    request=payload_scoring,
                    response={"fields": scoring_response['predictions'][0]['fields'], "values":scoring_response['predictions'][0]['values']},
                    response_time=460
                )])
        time.sleep(30)
        pl_records_count = self.wos_client.data_sets.get_records_count(self.payload_data_set_id)
        print("Number of records in the payload logging table: {}".format(pl_records_count))