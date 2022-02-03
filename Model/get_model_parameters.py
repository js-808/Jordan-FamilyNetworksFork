import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph_attributes import *
from scipy.stats import linregress
import pandas as pd
import pyarrow.feather as feather
import ast
import random


def find_distances_bio(g_num, graph_names):
    """Finds length of shortest 'biological' path between each marriage node in a given graph (i.e., path goes through parents)
    
    Parameters: g_num (int): graph number in list of graphs
 
    Returns: distances (list): list of union distances
             count (int): number of infinite union distances
    """
    print(graph_names[g_num])
    og_graph = graph_with_attributes(graph_names[g_num])
    
    # get all parts of graph
    stuff = separate_parts(graph_names[g_num],'A')

    # list of marriage edge tuples for the given graph
    marriages = stuff[1]

    # list of parent child edge tuples for the given graph
    children = stuff[2]

    distances = []
    count = 0

    # delete all marriage edges first
    for i in range(len(marriages)):

        # get parent nodes
        p1, p2 = marriages[i]

        # delete marriage edge
        og_graph.remove_edge(p1,p2)

    # go through all marriage pairs
    for i, m in enumerate(marriages):
        # copy graph to change temporarily
        g = og_graph.copy()

        # get parent nodes
        p1, p2 = m

        # find children of each parent node & delete children nodes from parent nodes
        for child in children:
            if child[0] == p1:
                g.remove_edge(p1,child[1])
            elif child[0] == p2:
                g.remove_edge(p2,child[1])
                
        try:
            # find shortest path between parent nodes
            path = nx.shortest_path(g, source=p1, target=p2)

            # record length of shortest path
            distances.append(len(path)-1)

        except nx.NetworkXNoPath:
            # count number of marriage nodes that don't have path between them
            count+=1

    return distances, count


def find_children(g_num, graph_names):
    og_graph = graph_with_attributes(graph_names[g_num])
    
    # get all parts of graph
    stuff = separate_parts(graph_names[g_num],'A')

    # list of marriage edge tuples for the given graph
    marriages = stuff[1]

    # list of parent child edge tuples for the given graph
    children = stuff[2]

    count = []
    
    # go through all marriage pairs
    for i, m in enumerate(marriages):

        # get parent nodes
        p1, p2 = m
        
        p1_list = set()
        p2_list = set()
        
        # find children of each parent node & count children node
        for child in children:
            if child[0] == p1:
                p1_list.add(child[1])
            elif child[0] == p2:
                p2_list.add(child[1])
    
        # union of p lists
        #p1_list |= p2_list
        
        #intersection of p lists
        c_list = p1_list.intersection(p2_list)
    
        count.append(len(c_list))
        
    return count


def get_some_parameters(g_num, name):
    """Get necessary parameters for model
    Parameters: 
                g_num (int): genealogical network number (i.e., the dataset you want to use for the model)
                name (str): name used to differentiate files (should be descriptive of chosen genealogical network)
    
    Returns:
                m_e (int): number of marriage edges in genealogical network
                inf (int): number of infinite distance marriages
                P (float): probability of marriage
                NCP (float): probability of non-connected marriage
    """
    
    # load all original sources
    graphs, graph_names = get_graphs_and_names()   # graphs is a list of all Kinsource graphs
                                                   # graph_names is a list of Kinsource graph file names 

    # genealogical network
    G = graphs[g_num]

    # get total number of nodes in network
    total = G.number_of_nodes()

    # Gives the number of males, females, unknown, marriage edges, parent-child edges in network
    attribs = count_attributes(G)
    
    # get number of marriage edges in network
    m_e = attribs[3]

    # get probability of marriage
    P = m_e*2/total
    
    # get all parts of graph
    stuff = separate_parts(graph_names[g_num],'A')

    # list of marriage edge tuples for the given network
    marriages = stuff[1]
    
    # list of parent-child edge tuples for the given network
    children = stuff[2]
    
    # get set of nodes that are married
    all_married = set()
    for pair in marriages:
        all_married.add(pair[0])
        all_married.add(pair[1])
    all_married  # set of all nodes that are married
    
    # get set of nodes that are children
    all_children = set()
    for pc in children:
        child = pc[1]
        all_children.add(child)
    all_children  # set of all nodes that are children

    # get nodes that are married but are not children
    not_children = all_married - all_children

    # get probability of non-connected marriage
    NCP = len(not_children)/len(all_married)
    
    # get list of finite distances and number of infinite distances
    finite_dis, inf_dis = find_distances_bio(g_num, graph_names)

    # Save list of finite distances as a txt file
    text_file = open("{}_distances.txt".format(name), "w")
    n = text_file.write(str(finite_dis))
    text_file.close()


    # get number of children from each union
    num_children = find_children(g_num, graph_names)

    # Save list of number of children as a txt file
    text_file = open("{}_children.txt".format(name), "w")
    n = text_file.write(str(num_children))
    text_file.close()

    return m_e, P, NCP, inf_dis
