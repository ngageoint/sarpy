import uuid
import json
import requests
import os
import glob

def get_atk_form_json(chain_filename):
    fp = open(chain_filename, 'r')
    chain_json = json.load(fp)
    fp.close()

    return chain_json


def call_atk_chain_standalone(atk_chains,  # type: AtkChains
                              chain_name,
                              api_key):
    status_key = uuid.uuid4().hex
    chain_json = atk_chains.get_chain_json(chain_name)

    chain = json.dumps(chain_json)

    request_json = {
        "api_key": api_key,
        "chain": chain,
        "output_type": "stdout",
        "status_key": status_key
    }

    r = requests.get(
        'http://localhost:5000/main/',
        request_json,
        verify=False
    )

    try:
        output = json.loads(r.content)['output_value']

    except (KeyError, ValueError):
        raise ValueError(str(r.__dict__))

    return status_key, output


def call_atk_chain(atk_chains,  # type: AtkChains
                   chain_name,
                   pass_params_in_mem=False):

    from algorithm_toolkit.atk import app

    status_key = uuid.uuid4().hex
    chain_json = atk_chains.get_chain_json(chain_name)

    if pass_params_in_mem:
        chain = 'from_global'
        app.config['CHAIN_DATA'][status_key] = chain_json
    else:
        chain = json.dumps(chain_json)

    request_json = {
        "api_key": app.config['API_KEY'],
        "chain": chain,
        "output_type": "stdout",
        "status_key": status_key
    }

    r = requests.get(
        'http://localhost:5000/main/',
        request_json,
        verify=False
    )

    try:
        output = json.loads(r.content)['output_value']

    except (KeyError, ValueError):
        raise ValueError(str(r.__dict__))

    return status_key, output


class AtkChains:
    def __init__(self, project_name):
        self.available_chains = {}
        form_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), project_name)
        form_path = os.path.join(form_path, "chain_forms/")
        form_filenames = glob.glob(os.path.join(form_path, '*.json'))

        for chain_filename in form_filenames:
            chain_json = get_atk_form_json(chain_filename)
            cname = chain_json['chain_name']
            self.available_chains[cname] = {
                'chain_json': chain_json,
                'chain_data': None
            }

    def get_available_chains(self):
        return sorted(list(self.available_chains.keys()))

    def get_chain_json(self, chain_name):
        return self.available_chains[chain_name]['chain_json']

    def get_chain_data(self, chain_name):
        return self.available_chains[chain_name]['chain_data']

    def set_chain_json(self, chain_name, chain_json):
        self.available_chains[chain_name]['chain_json'] = chain_json

    def set_chain_data(self, chain_name, chain_data):
        self.available_chains[chain_name]['chain_data'] = chain_data
