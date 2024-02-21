#OpenScale authentication

import json
from ibm_watson_machine_learning import APIClient


from ibm_watson_openscale import *
from ibm_watson_openscale.supporting_classes.enums import *
from ibm_watson_openscale.supporting_classes import *
import os
class WOSAuthenticator:
    def __init__(self):
        self.ibm_cloud = False

    def authenticate(self):
        if self.ibm_cloud==True:
            from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
            authenticator = IAMAuthenticator(apikey=os.environ['API_KEY'])
            wos_client = APIClient(authenticator=authenticator)
        else:
            from ibm_cloud_sdk_core.authenticators import CloudPakForDataAuthenticator

            # authenticator = IAMAuthenticator(username=self.credentials['username'], password=self.credentials['password'])
            
            authenticator = CloudPakForDataAuthenticator(
                    url=os.environ["WOS_URL"],
                    username=os.environ["USERNAME"],
                    apikey=os.environ["API_KEY"],
                    disable_ssl_verification=True
                )
            wos_client = APIClient(service_url=os.environ["CPD_URL"],authenticator=authenticator, service_instance_id=os.environ["WOS_INSTANCE_ID"])

                        
                        


        
        print(wos_client.version)
        return wos_client

###done#