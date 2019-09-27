import uuid
import json
import requests


def get_atk_form_json(chain_filename):
    fp = open(chain_filename, 'r')
    chain_json = json.load(fp)
    fp.close()

    return chain_json


def call_atk_chain(chain_json, api_key):
    status_key = uuid.uuid4().hex
    request_json = {
        "api_key": api_key,
        "chain": json.dumps(chain_json),
        "output_type": "stdout",
        "status_key": status_key
    }

    r = requests.get(
        'http://localhost:5000/main/',
        request_json,
        verify=False
    )

    try:
        result = json.loads(r.content)['output_value']
        print(result)

    except (KeyError, ValueError):
        raise ValueError(str(r.__dict__))
