from numpy import loadtxt, reshape
import json
from sys import getsizeof, exc_info

def serialize(filename, A = 1):
    '''
    Serialize numpy graph.
    Scheme:{"name":<graph>,
            "data":{
            "mat":[...],
            "shape":.}
            "weight":<A>}
    '''
    try:
        graphDat = loadtxt(filename)
        data = {"mat": graphDat.flatten().tolist(), "shape": graphDat.shape}
        jsonStr = json.dumps({"name": filename, "data": data, "weight": A})
        return jsonStr, str(round(getsizeof(jsonStr) / 10**6, 3)) + " MB"
    except Exception as e:
        print('😢 Goodbye World: Error on line {}'.format(exc_info()[-1].tb_lineno)," | ",type(e).__name__," | ",e)
        exit()

def deserialize(jsonData):
    '''
    jsonData needs to be consistent with source format
    Scheme:{"name":<graph>,
            "data":{
            "mat":[...],
            "shape":.}
            "weight":<A>}
    '''

    try:
        jsonData = json.loads(jsonData)
        #jsonData.pop("data") 🤦‍♂️dont activate this
        #quick & dumb way for format check - routes to KeyError
        dataDict = jsonData["data"]
        dataName = jsonData["name"]
        dataWeight = jsonData["weight"]
        return dataName, reshape(dataDict["mat"], dataDict["shape"]), dataWeight
    except Exception as e:
        print('😢 Goodbye World: Error on line {}'.format(exc_info()[-1].tb_lineno)," | ",type(e).__name__," | ",e)
        exit()
