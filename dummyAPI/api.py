import requests
import pandas as pd
from .jsonSerializer import serialize

def get(key):

    checkStatusUrl = 'https://ucp.unicen.smu.edu.sg/da_api/checkStatus/'
    getDataUrl = 'https://ucp.unicen.smu.edu.sg/da_api/getData/'
    makeRequest = requests.get(checkStatusUrl+key)
    if makeRequest.text == "complete":
        makeRequest = requests.get(getDataUrl+key)
        return convertJson2Pd(makeRequest.text)
    else:
        return makeRequest.text

def post(filename, A):

    executeDAUrl = 'https://ucp.unicen.smu.edu.sg/da_api/executeDA'
    jsonData, size = serialize(filename, A)
    r = requests.post(executeDAUrl, data=jsonData)

    return r.status_code, r.reason, r.text


def convertJson2Pd(txt):
    import json
    import numpy as np
    import pandas as pd

    raw_data = json.loads(txt)
    # list(raw_data[0].keys())[0]
    fields = list(raw_data[0].keys())[0].replace(' ','').split(',')
    _data = [list(line.values())[0].split(' ') for line in raw_data]
    _data = np.asarray(_data).astype(float)
    da_data = pd.DataFrame(data=_data, columns=fields)
    return da_data