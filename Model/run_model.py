from family_network_model import *
from matplotlib import pyplot as plt
import networkx as nx
import ast
import pickle
from get_model_parameters import *
import ast



# name of chosen network
name = 'tikopia_1930'  # CHANGE

# get data
with open('./UnionDistances/other_parameters.txt') as f:
    params = f.readline()
data_parameters = ast.literal_eval(params)
# save the following parameters:
m_e = data_parameters[name][0]  # number of marriage edges
P = data_parameters[name][1]    # probability of marriage
NCP = data_parameters[name][2]  # probability of nonconnected marriage

# get number of infinite distance unions
with open('./UnionDistances/infinite_distances.txt') as f:
    inf_distances = f.readline()
infinite_distances = ast.literal_eval(inf_distances)
# save number of infinite distance unions as a parameter
inf_dis = infinite_distances[name]

# marriage distance data of chosen network
with open('./UnionDistances/{}_distances.txt'.format(name)) as f:
    nx_dis = f.readline()
# network's marriage distances w/o infinite distances distribution
nx_distances = ast.literal_eval(nx_dis)

# number of children data of chosen network
with open('./ChildrenNumber/{}_children.txt'.format(name)) as f:
    nx_child = f.readline()
# network's number of children distribution
nx_children = ast.literal_eval(nx_child)



# initialize all parameters

n = 99    # CHANGE
gen = 10  # CHANGE
name = name + '_test1' # CHANGE

marriage_dist = nx_distances
children_dist = nx_children
p = P
ncp = NCP
infdis = round((inf_dis/m_e - (NCP/2))*m_e)


# run model
G, D, unions, children, infdis = human_family_network(n, gen, marriage_dist, p, ncp, infdis, children_dist, name)
