import numpy as np
from scipy import random
import string
import math

def gen_bounds():
    '''
    Get bounds for master gen
    '''

    x_l = random.randint(10, 90)
    x_h = random.randint(x_l+30, x_l+50)
    y_l = random.randint(10, 90)
    y_h = random.randint(y_l+30, y_l+50)

    return x_l, x_h, y_l, y_h

def gen_exponential(seedx, seedy):
    '''
    Generate exponential coordinates
    '''

    return random.exponential(seedx), random.exponential(seedy)

def gen_uniform(seedxl, seedxh, seedyl, seedyh):
    '''
    Generate uniform coordinates
    '''

    return int(random.uniform(seedxl, seedxh)), int(random.uniform(seedyl, seedyh))

def gen_master(x_l, x_h, y_l, y_h):
    '''
    Decide exponential or uniform
    '''

    if random.randint(1,2,1) > 1:
        return gen_exponential(random.uniform(x_l, x_h), random.uniform(y_l, y_h))
    else:
        return gen_uniform(x_l, x_h, y_l, y_h)

def gen_size():
    '''
    Generate data size
    '''

    return random.randint(20, 30)

def is_valid(coordinates, new_point, limit = 1):
    '''
    Is the new_point distant enough from others?
    limit? how to set - heuristic = 1
    '''

    for coord in coordinates:
        if np.linalg.norm(list((coord[0] - new_point[0], coord[1] - new_point[1]))) <= limit:
            return False
    return True

def gen_name(size = 7, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    '''
    Generate a random name for graph
    '''

    return ''.join(np.random.choice(list(chars)) for _ in range(size))

def gen_array(coords):
    '''
    Generate a numpy array from list of coords
    '''


    return np.array([[int(math.sqrt((coords[node1][0] - coords[node2][0]) ** 2 +
                                                       (coords[node1][1] - coords[node2][1]) ** 2) + .5)
                                         for node1 in range(coords.__len__())] for node2 in range(coords.__len__())])

def main(howMany=100):
    '''
    Generate 100 random fully connected graphs
    '''

    for i in range(howMany):
        graph = {}
        x_l, x_h, y_l, y_h = gen_bounds()
        size = gen_size()
        name = gen_name()
        graph[name, size] = {}
        j = 0
        while j < size:
            coord = gen_master(x_l, x_h, y_l, y_h)
            if is_valid(list(graph[name, size].values()), coord):
                graph[name, size][j] = coord
                j += 1
        for key in graph.keys():
            nparray = gen_array(list(graph[key].values()))
            nparray.dump(".\\graphs\\"+key[0]+"_"+str(key[1]))

main()