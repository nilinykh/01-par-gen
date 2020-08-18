'''configuring MTurk account given settings / credentials in config.ini'''

import configparser
import boto3
CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

class ConnectToMTurk():
    '''defines MTurk working environment'''

    def __init__(self, mturk):
        self.mturk = mturk

    create_hits_in_production = CONFIG.getboolean('environment', 'value')
    environments = {
        'production': {
            'endpoint_url': 'https://mturk-requester.us-east-1.amazonaws.com'
        },
        'sandbox': {
            'endpoint_url': 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
        },
    }
    mturk_environment = environments['production']['endpoint_url'] if\
                        create_hits_in_production else environments['sandbox']['endpoint_url']
    mturk = boto3.client(
        'mturk',
        aws_access_key_id=CONFIG['credentials']['id'],
        aws_secret_access_key=CONFIG['credentials']['key'],
        region_name='us-east-1',
        endpoint_url=mturk_environment
    )
