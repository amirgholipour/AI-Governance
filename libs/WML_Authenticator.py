#Watson Machine Learning authentication

import json
from ibm_watson_machine_learning import APIClient

import os
class WMLAuthenticator:
    def __init__(self):
        self.cp4d = True

    def authenticate(self):
        if self.cp4d==True:
            # authenticator = IAMAuthenticator(username=self.credentials['username'], password=self.credentials['password'])
            wml_credentials = {
                "url":os.environ["CPD_URL"],
                "username": os.environ["USERNAME"],
                #"password": os.environ["PASSWORD"],
                "apikey":os.environ["API_KEY"],
                "instance_id": "wml_local",
                "version" : "4.8" #If your env is CP4D 4.x.x then specify "4.x.x" instead of "4.5"  
            }
            wml_client = APIClient(wml_credentials)
            
        else:
            wml_credentials = {
                "apikey":os.environ['API_KEY_WX'],
                "url": 'https://us-south.ml.cloud.ibm.com' # 'https://' + location + '.ml.cloud.ibm.com'
            }
            wml_client = APIClient(wml_credentials)
            

        print(wml_client.version)
        wml_client.set.default_space(os.environ["SPACE_ID"])

        return wml_client
