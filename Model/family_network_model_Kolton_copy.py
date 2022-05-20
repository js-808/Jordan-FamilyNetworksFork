import networkx as nx
import random
import functools
import operator
import numpy as np
import ast
from scipy import stats
from sklearn.neighbors import KernelDensity as KDE
from scipy import interpolate
import itertools
import pickle
from time import time
from functools import wraps

# def timeit(func):
#     """
#     :param func: Decorated function
#     :return: Execution time for the decorated function
#     """
#
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start = time()
#         result = func(*args, **kwargs)
#         end = time()
#         print(f'{func.__name__} executed in {end - start:.4f} seconds')
#         return result
#
#     return wrapper
people = [k for k in range(10)]
prob_marry_immigrant = 0.2
prob_marry = 0.6
def kolton_add_marriage_edges(people, prob_marry_immigrant, prob_marry, D, indices):
    """
    people:  (list) of the current generation (IE those people elligible for
            marriage)
    prob_marry_immigrant: (float) the probablility that a given node will marry
            a immigrant (herein a person from outside the genealogical network,
            without comon ancestor and therefore at distance infinity from the
            nodes in the list 'people') (formerly 'ncp')
    prob_marry: (float) the probability that a given node will marry another
            node in people
    D: ((people x people) numpy array) indexed array of distance between nodes in people (siblings are distance 2)
    indices: (dictionary) maps node name (int) to index number in D (int)

    """
    people_set = set(people)  # for fast removal later
    # find the next 'name' to add to your set of people
    next_person = np.max(people) + 1
    # number of non-connected people to add
    num_immigrants = round(prob_marry_immigrant * len(people))  # m
    # marry off the immigrants at random to nodes in the current generation
    marry_strangers = np.random.choice(people, size=num_immigrants)
    unions = {(spouse, immigrant) for spouse, immigrant
                                    in zip(marry_strangers,
                                           range(next_person, next_person + num_immigrants))}
    # remove the married people from the pool #2ndManifesto
    people_set = people_set - set(marry_strangers)

    # get number of people to marry
    num_couples_to_marry = round(len(people_set)*prob_marry/2)
    # get all possible pairs of the still single nodes
    # rejecting possible parrings which have a common ancestor more recently
    # than allowed by marriage_probs (IE this is where we account that siblings
    # don't marry in most cultures (but still can such as in the tikopia_1930
    # family network))
    possible_couples = {(man, woman): D[indices[man]][indices[woman]]
                        for man, woman in itertools.combinations(people_set, 2)
                        if D[indices[man]][indices[woman]] > min(marriage_probs)}
    iter = 0
    while possible_couples and iter < num_couples_to_marry:

possible_couples
def add_marriage_edges(all_fam, all_unions, D, marriage_probs, p, ncp, n, infdis, indices):
    """Add marriage edges to a given generation
    Paramters:
                all_fam (list): list of all nodes in current generation--all eligible nodes for marrige.
                                (Flattened list of families)
                D (array): nxn matrix of distances between nodes
                marriage_probs (dict): dictionary of marriage distance probabilities.
                                        (e.g., value at key 5 is probability of distance of length 5)
                p (float): probability of marrying
                ncp (float): probability of a nonconnected marriage
                n (int): current number of nodes in network
                inf_dis (int): number of marriage pairs in data that have an infinite distance
    Returns:
                new_unions (list): list of edges that represent new unions between nodes
                no_unions (list): list of nodes with no unions
                all_unions (list): list of all unions since generation 0
                n (int): current number of nodes in network
                m (int): number of nonconnected nodes added to network

    """
    unions = []
    # number of non-connected people to add
    m = round(((ncp*len(all_fam))/(1-ncp))/2)
    infdis -= m
    # list of people not connected to family network
    nc_ppl = []
    print("all_fam: ", all_fam)
    print("indices: ", indices)
    max_ind = max(indices.values())
    for i in range(1, m+1):
        nc_ppl.append(n+i)
        indices[n+i] = max_ind + i

    # marry off non-connected people
    for nc in nc_ppl:
        spouse = random.choice(all_fam)
        unions.append((nc, spouse))
        all_fam.remove(spouse)
    #Shouldn't this be m + n - |nc|?  We just married off some people above...
    # IE don't we have that m couples got married in the previous for loop,
    # so that there are n-m eligible candidates left?
    # select how many of the n+m people get married based on probability of marriage
    # WRONG.  k = round(((n+1)*p)/2)
    k = round(((n-m+1)*p)/2)

    # UPDATED: moved this line down so that it wouldn't need to be undone when
    #          calculating k
    # update number of people in network
    n = n + m

    # get all possible pairs of rest of nodes
    all_pairs = list(itertools.combinations(all_fam, 2))

    # get all possible pairs of nodes that can marry
    poss_dis = {}
    #pair = (0,1)
    for pair in all_pairs.copy():
        # get distance of pair
        dis = D[indices[pair[0]]][indices[pair[1]]]

        # check that pair can marry (i.e., siblings can't marry)
        if dis < min(marriage_probs):
            all_pairs.remove(pair)

        # keep track of distances of all possible nodes to marry
        else:
            if dis >= 100:
                poss_dis[pair] = 100  # all infinite distances are '100'
            else:
                poss_dis[pair] = dis



    while (len(all_pairs) > 0) and (len(unions) <= k):
        # find probabilities of all possible distances--must update after creating each union
        dis_probs = []
        for d in poss_dis.values():
            dis_probs.append(marriage_probs[d])

        # normalize probabilities--must update after creating each union
        rel_probs = list(np.array(dis_probs)/sum(dis_probs))  # relative probability of distances

        # choose pair to marry based on relative probability of distances
        marry = random.choices(population=all_pairs, weights=rel_probs)[0]
        unions.append(marry)

        # remove all pairs that include one of the nodes of 'marry'
        for pair in all_pairs.copy():
            if (pair[0] == marry[0]) or (pair[0] == marry[1]) or (pair[1] == marry[0]) or (pair[1] == marry[1]):
                all_pairs.remove(pair)
                poss_dis.pop(pair)

    # keep track of nodes that didn't marry
    no_unions = list(set(all_fam) - set(functools.reduce(operator.iconcat, unions, ())))
    all_unions += unions

    return unions, no_unions, all_unions, n, m, infdis, indices


def add_children_edges(unions, n, child_probs, all_children, indices):
    """Add children edges to a given generation of unions
    Parameters: unions (list): unions formed in current generation
                n (int): current number of nodes in graph
                child_probs (dict): dictionary of number of children probabilities
                                    (e.g., value at key 5 is probability of having 5 children)
    Returns: child_edges (list): list of new parent-child edges to add to graph
             families (list): list of lists where each inner list represents siblings
             n (int): new number of nodes in graph after adding children
             all_children (list): list of number of children per union since generation 0

    """

    families = []

    child_edges = []
    union = (5,2)
    for union in unions:
        # how many children the union will have--based on children data distribution
        num = random.choices(population=list(child_probs.keys()), weights=list(child_probs.values()))[0]

        # add children
        if num != 0:
            # initialize list of children
            children = []
            max_ind = max(indices.values())
            # add children nodes to graph
            for c in range(num):
                # specify node number (label) to add
                n = n+1
                indices[n] = max_ind + c + 1
                children.append(n)
                # add edges from child to parents
                child_edges.append((union[0], n))
                child_edges.append((union[1], n))
            # keep track of families (IE which nodes are siblings and are thus ineligible marraige partners)
            families.append(children)

        # no children so add empty list
        else:
            families.append([])

        all_children.append(num)

    return child_edges, families, n, all_children, indices


def update_distances(D, n, unions, no_unions, families, indices):
    """Build a distance matrix that keeps track of how far away each node is from each other.
        Need to update distances after new nodes added to graph (i.e., after adding children)
    Parameters: D (array): "old" matrix of distances
                n (int): number of nodes currently in graph
                unions (list):
                no_unions (list):
                families (list):
                indices (dictionary): maps node name (an int) to index number (row and column number) in the current distance matrix D.
    Returns: D1 (array): "new" (updated) matrix of distances
    """
    # initialize new matrix
    num_children = len([child for fam in families for child in fam])
    D1 = np.zeros((num_children, num_children))
    new_indices = {child:k for k, child in enumerate([child for fam in families for child in fam])}
    # compute new distances
    unions
    u, union = 0, (5,4)
    other = (0,2)
    for u, union in enumerate(unions):
        u_children = families[u]

        for other in unions:
            if (union != other):
                o_children = families[unions.index(other)]

                # find all possible distances from union to other
                d1 = D[indices[union[0]]][indices[other[0]]]
                d2 = D[indices[union[1]]][indices[other[0]]]
                d3 = D[indices[union[0]]][indices[other[1]]]
                d4 = D[indices[union[1]]][indices[other[1]]]

                # compute distance between children of union and children of other
                d = min(d1, d2, d3, d4) + 2

                for uc in u_children:
                    for oc in o_children:
                        D1[new_indices[uc]][new_indices[oc]] = d
                        D1[new_indices[oc]][new_indices[uc]] = d

        # add immediate family distances
        for ch in u_children:
            # add sibling distances
            for c in u_children:
                if ch != c:
                    D1[new_indices[ch]][new_indices[c]] = 2
                    D1[new_indices[c]][new_indices[ch]] = 2

    return D1, new_indices


def toKDE(data, bandwidth, kernel='gaussian'):
    #data is a list of numbers that occur, and you want to find a KDE of their frequency.
    return KDE(kernel=kernel,bandwidth=bandwidth).fit(np.array(data).reshape(-1,1))


def get_probabilities(data, gen=0):
    """Create dictionary of probabilities based on real data distribution
    Parameters:
                data (list): data from a real family network
                gen (int): number of generations the network will grow (only need for marriage distribution)
    Returns:
                probs (dictionary): probabilities of all possible datapoints
                                    (e.g., probability at index 5 is probability of distance/children of length 5)

    """

    # changing bandwidth changes KDE
    bandwidth = 1

    # get kernel density of data
    kde = toKDE(data, bandwidth)
    x = np.arange(min(data)-1, max(data)+2, 1) # start and stop might need to change
    domain = x[:,np.newaxis]
    logs = kde.score_samples(domain)  # evaluate log density model on data
    y = np.exp(logs)  # get density model

    # fit spline (i.e., fit equation to density curve to then be able to integrate)
    spl = interpolate.InterpolatedUnivariateSpline(x,y)

    # create dictionary of probabilities by integrating under density curve
    probs = {}
    keys = set(data)
    for i in range(min(keys), max(keys)+gen+2):
        probs[i] = spl.integral(i-.5, i+5)

    return probs



n = 99
gen = 5
# marriage distance data of chosen network
name = "san_marino"
name = "anuta_1972"
name = "tikopia_1930"
with open('./UnionDistances/{}_distances.txt'.format(name)) as f:
    nx_dis = f.readline()
# network's marriage distances w/o infinite distances distribution
marriage_dist = ast.literal_eval(nx_dis)

# get number of infinite distance unions
with open('./UnionDistances/infinite_distances.txt') as f:
    inf_distances = f.readline()
infinite_distances = ast.literal_eval(inf_distances)
# save number of infinite distance unions as a parameter
infdis = infinite_distances[name]

# number of children data of chosen network
with open('./ChildrenNumber/{}_children.txt'.format(name)) as f:
    nx_child = f.readline()
children_dist = ast.literal_eval(nx_child)

# get data
with open('./UnionDistances/other_parameters.txt') as f:
    params = f.readline()
data_parameters = ast.literal_eval(params)
# save the following parameters:
m_e = data_parameters[name][0]  # number of marriage edges
p = data_parameters[name][1]    # probability of marriage
ncp = data_parameters[name][2]  # probability of nonconnected marriage
name
@timeit
def test(n):
    # fill in distances with data
    d = np.triu(np.random.choice(marriage_dist, size=(n+1,n+1)), k=1) # using marriage_dist instead of all_distances gives D all finite distances
    D = d + d.T
    return D
test(1000)



def human_family_network(n, gen, marriage_dist, p, ncp, infdis, children_dist, name):
    """Create an artifical human family network
    Parameters:
                n (int): number of nodes for initial graph minus 1
                          (e.g., to initialize a graph of 100 nodes, set n=99)
                gen (int): number of generations for network to grow for gen+1 total generations in network
                          (e.g., when gen=3, the network grows from generation 0 to generation 4)
                marriage_dist (list): data distribution from real family network of how "far" people married
                p (float): probability of marriage
                ncp (float): probability of marriage of nonconnected nodes
                infdis (int): number of infinite distance marriages
                children_dist (list): data distribution from a real family network of number of children per union
                name (str): name for prefix of saved files

    Returns: G (graph): contrived human family network
             D (array): matrix of distances between nodes in network
             all_unions (list): list of all marriage pairs in network
             all_children (list): list of number of children per union since generation 0

    """

    # generate empty graph
    G = nx.Graph()

    # add nodes to graph
    G.add_nodes_from([i for i in range(n+1)])

    # create lists needed for function
    families = []
    for node in G.nodes():
        families.append([node])
    all_fam = functools.reduce(operator.iconcat, families, [])
    all_unions = []
    all_children = []


    # initialize distance matrix of specified size
    # D = np.zeros((n+1,n+1))  # now created below.  no need to instantiate now.

    # create list of all possible distances (including infinite distances)
    all_distances = marriage_dist.copy() + [100 for i in range(infdis)]   # best number of infdis to add?

    # fill in distances with data
    d = np.triu(np.random.choice(marriage_dist, size=(n+1,n+1)), k=1) # using marriage_dist instead of all_distances gives D all finite distances
    D = d + d.T
    indices = {node:k for k, node in enumerate(range(n+1))}

    # get probabilities of possible distances to use in marriage function
    marriage_probs = get_probabilities(marriage_dist, gen=gen) # dictionary of probabilities of all finite distances
    marriage_probs[100] = (infdis/len(all_distances))/2  # include probability of infinite distance
    factor = 1.0/sum(marriage_probs.values())   # normalizing factor
    # normalize values for finite and infinite distances
    for k in marriage_probs:
        marriage_probs[k] = marriage_probs[k]*factor

    # get probabilities of possible number of children to use in children function
    child_probs = get_probabilities(children_dist)  # dictionary of probabilities of all possible number of children

    # add specified number of generations to network
    for i in range(gen+1):
        print('generation: ', i)

        # save output at each generation
        Gname = "{}_G{}.gpickle".format(name, i)   # save graph
        nx.write_gpickle(G, Gname)
        Dname = "{}_D{}.npy".format(name, i)   # save D
        np.save(Dname, D)
        indicesname = "{}_indices{}.npy".format(name, i)  # save indices
        np.save(indicesname, indices)
        Uname = "{}_U{}".format(name, i)   # save unions
        with open(Uname, 'wb') as fup:
            pickle.dump(all_unions, fup)
        Cname = "{}_C{}".format(name, i)   # save children
        with open(Cname, 'wb') as fcp:
            pickle.dump(all_children, fcp)

        # create unions between nodes to create next generation
        unions, no_unions, all_unions, n, m, infdis, indices = add_marriage_edges(all_fam, all_unions, D, marriage_probs, p, ncp, n, infdis, indices)
        G.add_edges_from(unions)

        oldn = n-m

        max_ind = max(indices.values())
        j=0
        oldn = len(indices) - m
        for j in range(m):
            # add non-connected ppl to distance matrix--infinte distances with everyone else
            r = np.ones((1,oldn+1+j))*100
            r[0,-1] = 0  # distance to self is 0
            c = np.ones((oldn+j,1))*100
            D = np.hstack((D, c))
            D = np.vstack((D, r))
            #indices[n+j+1] = max_ind + j + 1

        # add children to each union
        children, families, n, all_children, indices = add_children_edges(unions, n, child_probs, all_children, indices)
        G.add_edges_from(children)
        all_fam = functools.reduce(operator.iconcat, families, [])

        # update distances between nodes
        D, indices = update_distances(D, n, unions, no_unions, families, indices)


        # save output of last generation
        if i == gen:
            print("Last generation: ", i+1)
            Gname = "{}_G{}.gpickle".format(name, i+1)   # save graph
            nx.write_gpickle(G, Gname)
            Dname = "{}_D{}.npy".format(name, i+1)   # save D
            np.save(Dname, D)
            indicesname = "{}_indices{}.npy".format(name, i)  # save indices
            np.save(indicesname, indices)
            Uname = "{}_U{}".format(name, i+1)   # save unions
            with open(Uname, 'wb') as fup:
                pickle.dump(all_unions, fup)
            Cname = "{}_C{}".format(name, i+1)   # save children
            with open(Cname, 'wb') as fcp:
                pickle.dump(all_children, fcp)

    print(G.number_of_nodes())

    return G, D, all_unions, all_children, infdis
