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
from mathutils import Vector
import time
import random
import json
import ast
import pickle
import heapq
import os
import argparse

from edge_detection import *
from group import *

#---------------------------------------------------------------------------------------------#
def render_model_views(source_file, group_file, target_file, image_file, model_name, n_views=0, angle=0, resolution=512):
    
    # Load Object
    bpy.ops.import_mesh.stl(filepath=str(source_file))
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    model = bpy.context.object
  
    # Create object placement
    model.location = (0, 0, 0)
    model.scale = [5 / max(model.dimensions)] * 3

    # rotation angles
    if n_views < 5:
        axis = 1
    else:
        axis = 2

    model.rotation_euler = (0,0,0)
    
    if angle != pi:
        model.rotation_euler[axis] = angle
        if axis == 0:
            model.rotation_euler[1] = pi
        elif axis == 1:
            model.rotation_euler[2] = pi
        elif axis == 2:
            model.rotation_euler[0] = pi

    else:
        model.rotation_euler[axis] = angle

    if n_views == 0:
        model.rotation_euler = (0,0,0)
    #--------------------------------------#
    
    # add edges to model object
    me = model.data
    
    print('num vertices ', me.vertices)
    
    edge_pts, edges, normals, triangles, background_edge_pts, background_edges, background_triangles = edge_extraction(me, thresh=0.6)
  
    # Create Camera
    camera_object = bpy.data.objects["Camera"]
    camera_object.location = (10, 0, 2.5)
    camera_object.rotation_mode = 'XZY'
    camera_object.rotation_euler = Euler((-0.27, pi / 2, pi / 2))
    
    # Create Light
    light_object = bpy.data.objects["Light"]
    light_object.location = (10, 5, 7)
    light_object.data.energy = 10000

    # Adjust Scene
    bpy.ops.object.select_all(action='DESELECT')
    if "Cube" in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = "RGB"
    bpy.context.scene.render.filepath = str(pathlib.Path(str(image_file).format(n_views)).absolute())
    bpy.ops.render.render(write_still=True)

    # define ground truths
    # define cube
    # Finding 2d pixel locations 
    coords_dict = pixel_extraction(model, camera_object, me, edge_pts, background_edge_pts, limit = 0.05)
    
    background_edge_dict = background_classification(model, me, background_edges, background_triangles, coords_dict)
    edge_dict = contour_classification(model, me, edges, normals, triangles, background_edges, background_triangles, coords_dict)
    
    total_edge_dict = {**background_edge_dict, **edge_dict}

    # group edges
    n_dict, e_dict, normals_dict, mean_dict, heap = heap_process(total_edge_dict)
    final_groups, final_groups_dict = grouping(n_dict, e_dict, normals_dict, mean_dict, heap, initial_cost=3)
    final_groups, final_groups_dict = super_grouping(final_groups, n_dict, e_dict, final_cost=30)

    # plot-----------------------------------------------------------------
    obscuring = (0, 255, 255) # yellow
    concave = (255, 0, 0) # blue
    convex = (0, 255, 0) # green
    thickness = 3
    
    proj = cv2.imread(bpy.context.scene.render.filepath)
    plot = True
    if plot: 
        for idx, v in enumerate(total_edge_dict.values()):

            start_point, end_point, contour = v[0], v[1], v[2]
            if contour == 'obscuring':
                proj = cv2.line(proj, start_point, end_point, obscuring, thickness)
            if contour == 'concave':
                proj = cv2.line(proj, start_point, end_point, concave, thickness)
            if contour == 'convex':
                proj = cv2.line(proj, start_point, end_point, convex, thickness)  

    cv2.imwrite(target_file, proj)


    proj = cv2.imread(bpy.context.scene.render.filepath)
    if plot:
        for count, g in enumerate(final_groups):
            color = list(np.random.choice(range(256), size=3))
            color = [int(color[0]), int(color[1]), int(color[2])]
            for val in check_dim(g):
                proj = cv2.line(proj, val[0], val[1], tuple(color), thickness)

    cv2.imwrite(group_file, proj)
    
    bpy.data.objects[str(model_name)].select_set(True)
    bpy.ops.object.delete()

    bpy.ops.wm.read_factory_settings()
    
    return final_groups_dict

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def render_batch(model_list, group_dir, target_dir,  image_dir, n_views=1, resolution=512):

    model_dict = {}
        
    # iterate through each model in list
    for idx, model in enumerate(model_list):
        
        model_file = glob.glob(f"raw_meshes/{model}.*")
        if len(model_file) == 0:
            continue
        model_file = model_file[0]
        if model_file[-3:] != 'stl' or glob.glob(f"{target_dir}/{model}*"):
            continue

        # set up rotations - n_views
        for i in range(n_views):
            group_file = f"{group_dir}/{model}_{i}.png"
            target_file = f"{target_dir}/{model}_{i}.png"
            image_file = f"{image_dir}/{model}_{i}.png"
            angle = i * 2 * pi / n_views
            groups_dict = render_model_views(model_file, group_file, target_file, image_file, model, i, angle, resolution)

            if groups_dict != None:
                model_dict[str(model) + '_' + str(i)] = {}
                model_dict[str(model) + '_' + str(i)] = groups_dict


if __name__ == "__main__":

    # load data
    data = np.loadtxt("genus/genus.txt", int, delimiter=',')

    # create directories
    if not os.path.exists("output/gt_edges"):
        os.mkdir("output/gt_edges")
    if not os.path.exists("output/groups"):
        os.mkdir("output/groups")

    # run rendering
    render_batch(data, "output/groups", "output/gt_edges", "output/images", resolution=512)







