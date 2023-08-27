import sys
import bpy
import cv2
import glob
import math
from math import pi
import mathutils
from mathutils import Euler
import pathlib
import numpy as np
import bmesh
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils.bvhtree import BVHTree
#from pycocotools import mask 
#from skimage import measure
from mathutils import Vector
import time
import random
import json
import ast
import pickle
import heapq

#from edge_detection import *

# PATH
#/Applications/blender.app/Contents/Resources/2.93/python/bin

# return the normal vector provided edge
def normal(e):

    e = e[0]

    e_vec1 = (-(e[1][1] - e[0][1]), (e[1][0] - e[0][0]))
    e_vec2 = ((e[1][1] - e[0][1]), -(e[1][0] - e[0][0]))

    ev1 = e_vec1/ (np.linalg.norm(e_vec1))
    ev2 = e_vec2 / (np.linalg.norm(e_vec2))

    return ev1, ev2

# find normal curvature 
def normal_curvature(curve_list):
    curvature_list = curve_list.copy()

    # arrange normals to match correct directions
    for i in range(len(curvature_list) - 1):
        e_norms = curvature_list[i]
        n_norms = curvature_list[i+1]

        a1 = np.dot(e_norms[0], n_norms[0])
        a2 = np.dot(e_norms[0], n_norms[1])

        if a1 < a2:
            n_norms = (n_norms[1], n_norms[0])
            curvature_list[i+1] = n_norms

    return curvature_list

# find angle between two edges- using normals
def angle_between(ev1, ev2, nv1, nv2):

    one = np.arccos(np.clip(np.dot(ev1, nv1), -1, 1))
    two = np.arccos(np.clip(np.dot(ev1, nv2), -1, 1))
    three = np.arccos(np.clip(np.dot(ev2, nv1), -1, 1))
    four = np.arccos(np.clip(np.dot(ev2, nv2), -1, 1)) 

    angle = min(one, two , three, four)

    return np.degrees(angle)

# process before grouping - duplicates; neighboring edges
class EdgeNode:

    def __init__(self, key, value, attr):
        self.key = key # store pixel values of edges
        self.value = value  # store mean, standard deviation for each group
        self.attr = attr

    # compares the second value
    def __lt__(self, other):
        
        if len(self.attr) == 0 and len(other.attr) == 0:
            return sum(self.attr) < sum(other.attr)

        elif len(self.attr) == len(other.attr):
            return sum(self.attr)/len(self.attr) < sum(other.attr)/len(other.attr) 
        
        else:
            return len(self.attr) < len(other.attr) 
        
        
    def __str__(self):
        return str("{} : {} : {}".format(self.key, self.value, self.attr))



# create heap for priority to determine grouping
def heap_process(edge_dict):

    # intersections, neighbor degrees
    # try sorting the neighbor degreesx

    # FIND EACH EDGE's NEIGHBOR!
    # create dict of neighbors - based on pixel values
    e_dict = {}
    n_dict = {}
    normals_dict = {}
    mean_dict = {}

    for i, e1 in enumerate(edge_dict.keys()):
        e = edge_dict[str(e1)]
        key_edge = [e[0], e[1]]
        e_dict[str([key_edge])] = e[2]

        count = 0
        for j, e2 in enumerate(edge_dict.keys()):
            if i == j:
                continue

            n = edge_dict[str(e2)]

            if (e[0] == n[0] and e[1] == n[1]):
                continue

            elif (e[0] == n[0] or e[1] == n[0] or e[0] == n[1] or e[1] == n[1]) and count == 0:
                key_edge = [e[0], e[1]]
                val_edge = [n[0], n[1]]
                n_dict[str([key_edge])] = []
                n_dict[str([key_edge])].append(val_edge)
                count += 1

            elif (e[0] == n[0] or e[1] == n[0] or e[0] == n[1] or e[1] == n[1]) and count > 0:
                key_edge = [e[0], e[1]]
                val_edge = [n[0], n[1]]
                n_dict[str([key_edge])].append(val_edge)
                count += 1
         

    process_dict(n_dict, e_dict)


    # CREATE the HEAP - # edge pixels - mean, sigma - neighboring curvature for sorting
    heap = []
    neighbor_dict_values = list(n_dict.values()) 
    neighbor_dict_keys = list(n_dict.keys()) 
   
    for i in range(len(neighbor_dict_keys)):
        e = ast.literal_eval(neighbor_dict_keys[i])
        neighbors = neighbor_dict_values[i]

        curvatures = []
        for j in range(len(neighbors)):
            n = neighbors[j]

            ev1, ev2 = normal(e)
            nv1, nv2 = normal([n])
            angle = angle_between(ev1, ev2, nv1, nv2)

            curvatures.append(angle)

        #curvatures.sort(reverse=True)
        l = e[0]
        normals_dict[str(e)] = normal(e)
        mean_dict[str(e)] = normal(e)
        heapq.heappush(heap, EdgeNode(e, normal(e), curvatures))


    return n_dict, e_dict, normals_dict, mean_dict, heap


def check_dim(n):
    check = len(np.array(n, dtype="object").shape)

    if check == 2:
        return [n]

    else:
        return n



# group smaller edges ------------------------------------------------------------------------#
def grouping(n_dict, e_dict, normals_dict, mean_dict, heap, initial_cost=3):


    groups = []

    cost = initial_cost # establish a cost for adding - keep low for post processing at decision boundary - 15
    # iterate through each edge, and see if it needs to be added
    neighbor_dict_values = list(n_dict.values()) 
    neighbor_dict_keys = list(n_dict.keys()) 

    while heap:
        h = heapq.heappop(heap)
        l = h.key.copy()

        # skip things on heap that are not in any dicts
        if not str(l) in normals_dict.keys():
            continue

        # make sure neighbors are unique
        neighbors = []
        [neighbors.append(x) for x in n_dict[str(l)] if x not in neighbors]

 
        keys = []
        values = []
        attrs = []
        new_neighbors = []
        other_neighbors = []
        save_neighbors = []

        normal_values = []
        if isinstance(normals_dict[str(h.key)], list):
            for add in normals_dict[str(h.key)]:
                normal_values.append(add)
        else:
            normal_values.append(normals_dict[str(h.key)])

        # iterate through each neighbor
        for i in range(len(neighbors)):

            n = neighbors[i].copy() # corresponding neighbor
            n_dim = check_dim(n)

            # use neighbor of neighbors to also compare to the cost
            n_neighbors = n_dict[str(n_dim)]
            one_neighbors = [n2 for n2 in n_neighbors if check_dim(n2) != check_dim(h.key) and n2 not in check_dim(neighbors)]
            angle = [angle_between(h.value[0], h.value[1], mean_dict[str(check_dim(o))][0], mean_dict[str(check_dim(o))][1]) for o in one_neighbors]

            # find curvature
            attr = angle_between(h.value[0], h.value[1], mean_dict[str(check_dim(n))][0], mean_dict[str(check_dim(n))][1])

            # compare to cost - for adding, as well as edge type
            if (attr < cost) and e_dict[str(l)] == e_dict[str(n_dim)]:
                # determine keys to add
                keys.append(n)
                save_neighbors.append(n)
                
                # compute new mean curvature
                if isinstance(normals_dict[str(n_dim)], list):
                    for add in normals_dict[str(n_dim)]:
                        normal_values.append(add)
                else:
                    normal_values.append(normals_dict[str(n_dim)])

                # find neighbors and compute new curvatures
                for one in one_neighbors:
                    new_neighbors.append(one)
               
            else:
                # add in other neighbors that do not meet threshold requirement
                other_neighbors.append(n)
      
        # stopping criterion - no more to add to group
        if len(save_neighbors) == 0 and len(attrs) == 0:
            groups.append(h.key)
            continue


        # update neighbors dict, normals dict, e dict with new super node
        check = len(np.array(keys, dtype="object").shape)
        name = h.key.copy()

        # name includes initial heap
        for idx, n_one in enumerate(keys):
            for val in check_dim(n_one):
                name.append(val)

        # compute mean for group
        add_normals = normal_curvature(normal_values)
        add_n1 = add_normals[0][0].copy()
        add_n2 = add_normals[0][1].copy()

        for j in range(1, len(add_normals)):
            add_n1 += add_normals[j][0]
            add_n2 += add_normals[j][1]
  
        add_n1 = (add_n1[0]/len(add_normals), add_n1[1]/len(add_normals))
        add_n2 = (add_n2[0]/len(add_normals), add_n2[1]/len(add_normals))
        mean = (add_n1, add_n2)

        # update the dicts with new nodes
        for new in new_neighbors:
            for k, idx in enumerate(n_dict[str(check_dim(new))]):

                delete = True
                for ls in check_dim(idx):
                    if ls not in name:
                        delete = False
                        break

                if delete:
                    n_dict[str(check_dim(new))].remove(idx)
                    n_dict[str(check_dim(new))].insert(k, name)

            res = []
            [res.append(x) for x in n_dict[str(check_dim(new))] if x not in res]
            n_dict[str(check_dim(new))] = res
            
            # don't double count 
            if new in other_neighbors:
                other_neighbors.remove(new)


        # if nodes weren't combined, still adjust dict
        for other in other_neighbors:
            for k, idx in enumerate(n_dict[str(check_dim(other))]):
              
                delete = True
                for ls in check_dim(idx):
                    if ls not in name:
                        delete = False
                        break

                if delete:
                    n_dict[str(check_dim(other))].remove(idx)
                    n_dict[str(check_dim(other))].insert(k, name)


            res = []
            [res.append(x) for x in n_dict[str(check_dim(other))] if x not in res]
            n_dict[str(check_dim(other))] = res
        
        # update the dicts for the new key    
        n_dict[str(name)] = new_neighbors + other_neighbors
        mean_dict[str(name)] = mean
        normals_dict[str(name)] = add_normals
        e_dict[str(name)] = e_dict[str(l)]


        # update key
        one_curve = [angle_between(h.value[0], h.value[1], mean_dict[str(check_dim(o))][0], mean_dict[str(check_dim(o))][1]) for o in n_dict[str(name)]]

        for a in one_curve:
            attrs.append(a)

        h.attr = attrs
        h.key = name.copy()
        h.value = mean

        # remove the nodes that have been combined ----
        del n_dict[str(l)]
        del e_dict[str(l)]
        del normals_dict[str(l)]
        del mean_dict[str(l)]

        for save in save_neighbors:
            save = check_dim(save)
            del n_dict[str(save)]
            del e_dict[str(save)]
            del normals_dict[str(save)]
            del mean_dict[str(save)]

        # put node back on queue
        heapq.heappush(heap, h)


    # final groups dict
    groups_dict = {}
    
    for i, g in enumerate(groups):
        n_dict[str(g)] = list(n_dict[str(g)])
        e_dict[str(g)] = list(e_dict[str(g)])
        groups_dict['group ' + str(i+1)] = []
        groups_dict['group ' + str(i+1)].append(g)
        groups_dict['group ' + str(i+1)].append(e_dict[str(g)])
    

    return groups, groups_dict
    


def process_dict(n_dict, e_dict):
    # process n dict - duplicates and reverse start and end points
    for n in n_dict:
        new_dict = []
        for o in n_dict[str(n)]:
            if o not in new_dict:
                new_dict.append(o) 

        n_dict[str(n)] = new_dict


    for n in list(n_dict.keys()):

        if n not in n_dict:
            continue

        # remove flipped start and end points
        n_edge = ast.literal_eval(n)
        to_delete_idx = []
        for j, o in enumerate(n_dict[str(n)]):
            if o[0] == n_edge[0][1] and o[1] == n_edge[0][0]:
                to_delete_idx.append(j)
            if o[0] == n_edge[0][0] and o[1] == n_edge[0][1]:
                to_delete_idx.append(j)
                
        to_delete_idx = sorted(to_delete_idx, reverse=True)
        to_delete_values = []

        for elem in to_delete_idx:
            to_delete_values.append(n_dict[str(n)][elem])
            n_dict[str(n)].pop(elem)


        to_delete_idx = []
        for o in n_dict[str(n)]:
            for p in n_dict[str(check_dim(o))]:
                if p in to_delete_values:
                    n_dict[str(check_dim(o))].remove(p)


        for val in to_delete_values:
            del n_dict[str(check_dim(val))]
            del e_dict[str(check_dim(val))]



# group after initial grouping
def super_grouping(groups, n_dict, e_dict, final_cost=30):
    
    for n in list(n_dict.keys()):
        # remove duplicates
        new_dict = []
        for o in n_dict[str(n)]:
            if check_dim(o) not in new_dict:
                new_dict.append(check_dim(o)) 

        n_dict[str(n)] = new_dict
    


    # initially process groups
    for idx in range(len(groups)):
        g_orig = groups[idx].copy()
    
        g_new = group_process(g_orig) # to help find correct start and end points of the super node   
        g_new = group_combine(g_new)
        g_new = sort_name(g_new)


        groups[idx] = g_new

        n_dict[str(g_new)] = n_dict[str(g_orig)]
        e_dict[str(g_new)] = e_dict[str(g_orig)]


        if g_new != g_orig:
            del n_dict[str(g_orig)]
            del e_dict[str(g_orig)]
        
        # fix the neighbors of dict
        for i in range(len(n_dict[str(g_new)])):
            n = n_dict[str(g_new)][i]
            if type(n[0]) is tuple:
                n = sorted(n)

            else:
                n = sort_name(group_combine(group_process(check_dim(n))))


            n_dict[str(g_new)][i] = n

        # exception case
        if g_new in n_dict[str(g_new)]:
            n_dict[str(g_new)].remove(g_new)


    for n in list(n_dict.keys()):
        # remove duplicates
        new_dict = []
        for o in n_dict[str(n)]:
            if check_dim(o) not in new_dict:
                new_dict.append(check_dim(o)) 

        n_dict[str(n)] = new_dict

    # determine additional grouping
    group_endpoints = {}
    count_endpoints = {}
    for g in groups:
        # make dict for each group's start and end point
        endpoint1 = g[0][0]
        endpoint2 = g[len(g)-1][1]

        node_dict, count_dict = {}, {}
        node_dict[str(endpoint1)] = []  # need this to keep track of neighboring nodes
        node_dict[str(endpoint2)] = []
        count_dict[str(endpoint1)] = 0 # counts number of neighbors to node
        count_dict[str(endpoint2)] = 0

        for n in n_dict[str(g)]:
            n_dim = check_dim(n)
            for z in n_dim:
                if endpoint1 in z:
                    node_dict[str(endpoint1)].append(n_dim)
                    count_dict[str(endpoint1)] += 1
                if endpoint2 in z:
                    node_dict[str(endpoint2)].append(n_dim)
                    count_dict[str(endpoint2)] += 1
        

        group_endpoints[str(g)] = node_dict
        count_endpoints[str(g)] = count_dict
    

    # dict - {group: {endpoint1: [...], endpoint2: [...]}}
    # now, iterate through each end point and check the number of neighbors that end point has
    # use heap
    heap = []
    for g in group_endpoints:
        g = ast.literal_eval(g)
        heapq.heappush(heap, g)

    final_groups = []
    while heap:
        g = heapq.heappop(heap)

        # skip things on heap that are not in any dicts
        if not str(g) in n_dict.keys():
            continue

        endpoint1 = g[0][0]
        endpoint2 = g[len(g)-1][1]

        g_new = g.copy()

        if len(group_endpoints[str(g)][str(endpoint1)]) == 0 and len(group_endpoints[str(g)][str(endpoint2)]) == 0:
            g_new = g.copy()

        # check number of neighbors
        elif count_endpoints[str(g)][str(endpoint1)] == 1 and count_endpoints[str(g)][str(endpoint2)] == 1:
            neighbors_endpoint1 = group_endpoints[str(g)][str(endpoint1)][0]
            neighbors_endpoint2 = group_endpoints[str(g)][str(endpoint2)][0] #[0] just due to dimensions issue
            if neighbors_endpoint1 == neighbors_endpoint2:
                g_new = merge_supernode(g, neighbors_endpoint1, group_endpoints, count_endpoints, n_dict, e_dict, final_cost)
            else:
                g_new = merge_supernode(g, neighbors_endpoint1, group_endpoints, count_endpoints, n_dict, e_dict, final_cost)
                g_new = merge_supernode(g_new, neighbors_endpoint2, group_endpoints, count_endpoints, n_dict, e_dict, final_cost)

        # check number of neighbors
        elif count_endpoints[str(g)][str(endpoint1)] == 1:
            neighbors_endpoint1 = group_endpoints[str(g)][str(endpoint1)][0]
            g_new = merge_supernode(g, neighbors_endpoint1, group_endpoints, count_endpoints, n_dict, e_dict, final_cost)

        elif count_endpoints[str(g)][str(endpoint2)] == 1:
            neighbors_endpoint2 = group_endpoints[str(g)][str(endpoint2)][0] #[0] just due to dimensions issue
            g_new = merge_supernode(g, neighbors_endpoint2, group_endpoints, count_endpoints, n_dict, e_dict, final_cost)


        # group is processed already
        if g == g_new:
            if len(g_new) == 1:
                g_copy = g_new.copy()[0]
                dif = abs(g_copy[0][0] - g_copy[1][0]) + abs(g_copy[0][1] - g_copy[1][1])
                if dif > 2:
                    final_groups.append(g_new)

            else:
                final_groups.append(g_new)

            continue


        heapq.heappush(heap, g_new)


    final_groups_dict = {}
    for i, g in enumerate(final_groups):
        final_groups_dict['group ' + str(i+1)] = []
        final_groups_dict['group ' + str(i+1)].append(g)
        final_groups_dict['group ' + str(i+1)].append(e_dict[str(g)])

    return final_groups, final_groups_dict



# decision criteria
def decision_boundary(group, neighbor, final_cost, group_endpoints):
    group_normal = normal(group)
    neighbor_normal = normal(neighbor)

    a = list(group_endpoints[str(group)].keys())
    b = list(group_endpoints[str(neighbor)].keys())

    if len(list(set(a) & set(b))) == 0:
        return False

    intersect = list(set(a) & set(b))[0]

    if len(group_endpoints[str(group)][intersect]) == 0:
        return False

    if len(group_endpoints[str(neighbor)][intersect]) == 0:
        return False

    supernode_group = group_endpoints[str(group)][intersect][0]
    supernode_neighbor = group_endpoints[str(neighbor)][intersect][0]

    decision = False

    angle = final_cost + 1
    angles = []
    angles.append(angle)
    for g in supernode_group:
        for n in supernode_neighbor:
            g = check_dim(g)
            n = check_dim(n)
            
            # long segments break apart
            z = g[0]
            dif1 = abs(z[0][0] - z[1][0]) + abs(z[0][1] - z[1][1])

            y = n[0]
            dif2 = abs(y[0][0] - y[1][0]) + abs(y[0][1] - y[1][1])
            
            if dif1 > 100 or dif2 > 100:
                continue

            if g[0][0] == n[0][0] or g[0][0] == n[0][1] or g[0][1] == n[0][0] or g[0][1] == n[0][1]:
                group_normal1 = normal(g)
                neighbor_normal1 = normal(n)
                angle = angle_between(group_normal1[0], group_normal1[1], neighbor_normal1[0], neighbor_normal1[1])
                angles.append(angle)
            

    for a in angles:
        if a < final_cost:
            return True

    return False
    #return angle < final_cost


def merge_supernode(group, neighbor, group_endpoints, count_endpoints, n_dict, e_dict, final_cost):

    # make sure you process neighbors correctly
    neighbors = []
    [neighbors.append(x) for x in n_dict[str(group)] if x not in neighbors]

    # find new neighbors
    n_neighbors = n_dict[str(neighbor)]
    one_neighbors = [n2 for n2 in n_neighbors if check_dim(n2) != check_dim(group) and n2 not in check_dim(neighbors)]

    two_neighbors = [n2 for n2 in neighbors if check_dim(n2) != check_dim(neighbor)]
    new_neighbors = []
    other_neighbors = []
    save_neighbors = []

    # find decision
    decision = decision_boundary(group, neighbor, final_cost, group_endpoints)

    # neighbors for name 
    keys = []

    if decision and e_dict[str(group)] == e_dict[str(neighbor)]:

        keys.append(neighbor)
        save_neighbors.append(neighbor)

        # find the neighbors
        for one in one_neighbors:
            new_neighbors.append(one)

        for two in two_neighbors:
            new_neighbors.append(two)
      
    else:
        return group


    # name of group
    name = group.copy()
    for idx, n_one in enumerate(keys):
        for val in check_dim(n_one):
            name.append(val.copy())
 
    name = sort_name(name)

    # new end points
    new_endpoint1 = name[0][0]
    new_endpoint2 = name[len(name)-1][1]

    # update the dicts with new nodes
    for new in new_neighbors:
        for k, idx in enumerate(n_dict[str(check_dim(new))]):

            delete = True
            for ls in check_dim(idx):
                ls_reverse = ls.copy()
                ls_reverse = [ls_reverse[1], ls_reverse[0]]

                if ls not in name and ls_reverse not in name:
                    delete = False
                    break


            if delete:
                n_dict[str(check_dim(new))].remove(idx)
                n_dict[str(check_dim(new))].insert(k, name)

                if str(new_endpoint1) in group_endpoints[str(check_dim(new))]:
                    if check_dim(idx) in group_endpoints[str(check_dim(new))][str(new_endpoint1)]:
                        index = group_endpoints[str(check_dim(new))][str(new_endpoint1)].index(check_dim(idx))
                        group_endpoints[str(check_dim(new))][str(new_endpoint1)].remove(check_dim(idx))
                        group_endpoints[str(check_dim(new))][str(new_endpoint1)].insert(index, name)

                if str(new_endpoint2) in group_endpoints[str(check_dim(new))]:
                    if check_dim(idx) in group_endpoints[str(check_dim(new))][str(new_endpoint2)]:
                        index = group_endpoints[str(check_dim(new))][str(new_endpoint2)].index(check_dim(idx))
                        group_endpoints[str(check_dim(new))][str(new_endpoint2)].remove(check_dim(idx))
                        group_endpoints[str(check_dim(new))][str(new_endpoint2)].insert(index, name)


        res = []
        [res.append(x) for x in n_dict[str(check_dim(new))] if x not in res]
        n_dict[str(check_dim(new))] = res


    # adjust the dicts
    if name != group:
        e_dict[str(name)] = e_dict[str(group)]
        n_dict[str(name)] = new_neighbors 

        del n_dict[str(group)]
        del e_dict[str(group)]
       
        for save in save_neighbors:
            save = check_dim(save)
            del n_dict[str(save)]
            del e_dict[str(save)]


    # adjust group endpoints dict and count dict
    node_dict, count_dict = {}, {}
    node_dict[str(new_endpoint1)] = []  # need this to keep track of neighboring nodes
    node_dict[str(new_endpoint2)] = []
    count_dict[str(new_endpoint1)] = 0 # counts number of neighbors to node
    count_dict[str(new_endpoint2)] = 0


    final_neighbors = []
    for n in n_dict[str(name)]:
        n_dim = check_dim(n)
        for z in n_dim:
            
            if new_endpoint1 in z:
                node_dict[str(new_endpoint1)].append(n_dim)
                count_dict[str(new_endpoint1)] += 1

            if new_endpoint2 in z:
                node_dict[str(new_endpoint2)].append(n_dim)
                count_dict[str(new_endpoint2)] += 1
            
            final_neighbors.append(z)


    if name != group:
        group_endpoints[str(name)] = node_dict
        count_endpoints[str(name)] = count_dict
        del group_endpoints[str(group)]
        del count_endpoints[str(group)]

        for save in save_neighbors:
            save = check_dim(save)
            del group_endpoints[str(save)]
            del count_endpoints[str(save)]


        # update dict counts
        for new in new_neighbors:
            if str(new_endpoint1) in group_endpoints[str(check_dim(new))]:
                count_endpoints[str(check_dim(new))][str(new_endpoint1)] = count_endpoints[str(check_dim(name))][str(new_endpoint1)]

            if str(new_endpoint2) in group_endpoints[str(check_dim(new))]:
                count_endpoints[str(check_dim(new))][str(new_endpoint2)] = count_endpoints[str(check_dim(name))][str(new_endpoint2)]


    return name


# just sorts the groups and individual edges
def group_process(g):

    g_copy = g.copy()
    # order the group
    for i in range(len(g_copy)):
        g_copy[i] = sorted(g_copy[i])

    g_copy = sorted(g_copy)

    return g_copy


# combine the edges with same slope and intersect
def group_combine(g, flag = False): # may need to worry about discontinuous junctions here - have to fix???

    for i in range(len(g)):
        to_delete = []

        for j in range(i+1, len(g)):
            
            norm1 = normal(check_dim(g[i]))
            norm2 = normal(check_dim(g[j]))
            angle = angle_between(norm1[0], norm1[1], norm2[0], norm2[1])

            if angle == 0:
                intersect = group_intersect(g[i], g[j])
                if len(intersect) != 0:
                    minimum = min(g[i][0], g[j][0])
                    maximum = max(g[i][1], g[j][1])

                    to_delete.append(j)
                    g[i] = [minimum, maximum]
                    
        to_delete = sorted(to_delete, reverse=True)
        
        for elem in to_delete:
            del g[elem]           
        
    return g


# see if two groups intersect, with equal slopes
def group_intersect(group1, group2):

    pts1 = pt_sets(group1)
    pts2 = pt_sets(group2)
    intersection = list(set(pts1) & set(pts2))

    return intersection


def pt_sets(group1):

    if (group1[1][1] - group1[0][1]) == 0:
        x_inter = 1
        y_inter = 0

    elif (group1[1][0] - group1[0][0]) == 0:
        x_inter = 0
        y_inter = 1

    elif (group1[1][1] - group1[0][1])/(group1[1][0] - group1[0][0]) < 1:
        slope = (group1[1][1] - group1[0][1])/(group1[1][0] - group1[0][0])
        x_inter = 1/slope
        y_inter = 1

    elif (group1[1][1] - group1[0][1])/(group1[1][0] - group1[0][0]) >= 1:
        slope = (group1[1][1] - group1[0][1])/(group1[1][0] - group1[0][0])
        x_inter = 1
        y_inter = slope


    # organize the list
    if x_inter == 0 and y_inter != 0:
        Y = np.arange(group1[0][1], group1[1][1], y_inter, dtype=int)
        X = group1[0][0] * np.ones(len(Y), dtype=int)

    elif x_inter != 0 and y_inter == 0:
        X = np.arange(group1[0][0], group1[1][0], x_inter, dtype=int)
        Y = group1[0][1] * np.ones(len(X), dtype=int)

    else:
        X = np.arange(group1[0][0], group1[1][0], x_inter, dtype=int)
        Y = np.arange(group1[0][1], group1[1][1], y_inter, dtype=int)
        length = min(len(X), len(Y))
        X = X[0:length]
        Y = Y[0:length]


    pts = []

    for i in range(len(X)):
        pts.append((X[i], Y[i]))

    pts.append((group1[1][0], group1[1][1]))

    
    return pts


# sort full name
def sort_name(old_name):

    new_name = []
    new_name.append(old_name[0])

    # remove first edge 
    del old_name[0]

    # forward pass
    while find_neighbor_forward(new_name, old_name):
        added_edge = find_neighbor_forward(new_name, old_name)
        combine_edge = new_name[len(new_name) - 1]

        if added_edge[0] == combine_edge[1]:
            del old_name[old_name.index(added_edge)]
            new_name.append(added_edge)

        elif added_edge[1] == combine_edge[1]:
            del old_name[old_name.index(added_edge)]
            temp = added_edge.copy()
            added_edge[0] = temp[1]
            added_edge[1] = temp[0]
            new_name.append(added_edge)


    # backward pass
    while find_neighbor_backward(new_name, old_name):
        added_edge = find_neighbor_backward(new_name, old_name)
        combine_edge = new_name[0]

        if added_edge[1] == combine_edge[0]:
            del old_name[old_name.index(added_edge)]
            new_name.insert(0, added_edge)

        elif added_edge[0] == combine_edge[0]:
            del old_name[old_name.index(added_edge)]
            temp = added_edge.copy()
            added_edge[0] = temp[1]
            added_edge[1] = temp[0]
            new_name.insert(0, added_edge)

    return new_name


def find_neighbor_backward(new_name, old_name):

    endpoint = new_name[0][0]
    for i in range(len(old_name)):
        if endpoint in old_name[i]:
            return old_name[i]


def find_neighbor_forward(new_name, old_name):

    endpoint = new_name[len(new_name)-1][1]
    for i in range(len(old_name)):
        if endpoint in old_name[i]:
            return old_name[i]

