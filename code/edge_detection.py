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

# PATH
#/Applications/blender.app/Contents/Resources/2.93/python/bin
    

# POINTS ON LINE-----------------------------------------------------------------------------------#
def points_on_line(a_b, c, N, endpoint):
    
    if a_b[0] != c[0] and a_b[1] != c[1] and a_b[2] != c[2]:
        x = np.linspace(a_b[0], c[0], N, endpoint)
        y = ((x - a_b[0])/(c[0] - a_b[0])) * (c[1] - a_b[1]) + a_b[1]
        z = ((y - a_b[1])/(c[1] - a_b[1])) * (c[2] - a_b[2]) + a_b[2]
    
    elif a_b[0] == c[0] and a_b[1] != c[1] and a_b[2] != c[2]:
        x = np.linspace(a_b[0], c[0], N, endpoint)
        y = np.linspace(a_b[1], c[1], N, endpoint)
        z = ((y - a_b[1])/(c[1] - a_b[1])) * (c[2] - a_b[2]) + a_b[2]
    
    elif a_b[0] != c[0] and a_b[1] == c[1] and a_b[2] != c[2]:
        x = np.linspace(a_b[0], c[0], N, endpoint)
        y = np.linspace(a_b[1], c[1], N, endpoint)
        z = ((x - a_b[0])/(c[0] - a_b[0])) * (c[2] - a_b[2]) + a_b[2]
   
    elif a_b[0] != c[0] and a_b[1] != c[1] and a_b[2] == c[2]:
        x = np.linspace(a_b[0], c[0], N, endpoint)
        y = ((x - a_b[0])/(c[0] - a_b[0])) * (c[1] - a_b[1]) + a_b[1]
        z = np.linspace(a_b[2], c[2], N, endpoint)
        
    else:
        x = np.linspace(a_b[0], c[0], N, endpoint)
        y = np.linspace(a_b[1], c[1], N, endpoint)
        z = np.linspace(a_b[2], c[2], N, endpoint)
    
    # add vector if it is actually within the triangle
    vectors = []
    for i in range(1, N):
        pt = mathutils.Vector((x[i], y[i], z[i]))
        vectors.append(pt)
    
    return vectors
    
# TRIANGLE TYPE ----------------------------------------------------------------------------------#
def triangle_type(model, me, v):
    
    a = me.vertices[v[0]].co
    b = me.vertices[v[1]].co
    c = me.vertices[v[2]].co
    
    # find points from each edge's mid point to the other vertex (e.g. a and b's midpoint to c)
    A = points_on_line((a+b)/2, c, 25, endpoint=False)
    B = points_on_line((a+c)/2, b, 25, endpoint=False)
    C = points_on_line((b+c)/2, a, 25, endpoint=False)
    
    # make into one list
    A.extend(B)
    A.extend(C)
    
    # count number of rays that are visible
    count = 0
    for l in A:
        if is_Visible(model, l, limit=1e-04, add_box=False)[0]:
            count += 1

    if count <= 3:
        return 'occluding'
    else:
        return 'visible'
    
    
# classify non-obscure edges as concave or convex-----------------------------------------
def convexity(model, me, edge, triangles, normals):
    
    # 1. find centroid of triangle A
    triangles1 = triangles[0]
    coords_1a = me.vertices[triangles1[0]].co
    coords_1b = me.vertices[triangles1[1]].co
    coords_1c = me.vertices[triangles1[2]].co
    centroid1 = (coords_1a + coords_1b + coords_1c)/3
    #x1 = mathutils.geometry.intersect_point_tri(centroid1, coords_1a , coords_1b, coords_1c)
    x1 = centroid1
    #co1 = x1 @ model.matrix_world
    
    # 2. find centroid of triangle b
    triangles2 = triangles[1]
    coords_2a = me.vertices[triangles2[0]].co
    coords_2b = me.vertices[triangles2[1]].co
    coords_2c = me.vertices[triangles2[2]].co
    centroid2 = (coords_2a + coords_2b + coords_2c)/3
    #x2 = mathutils.geometry.intersect_point_tri(centroid2, coords_2a , coords_2b, coords_2c)
    x2 = centroid2
    #co2 = x2 @ model.matrix_world

    # 3. find surface normal for both
    n1 = normals[0] 
    n2 = normals[1] 
    
    #4. define a vector x1 - x2, using centroids
    d = x1 - x2
    d = d/(np.linalg.norm(d) + 1e-10)
 
    #5. find alpha1 and alpha 2 condition
    cond = np.dot(n1, d) - np.dot(n2, d)

    if cond >= 0:
        return "convex"
    else:
        return "concave"
    

# EXTRACT THE EDGES ------------------------------------------------------------------------------#
def edge_extraction(me, thresh = 0.2):
    
    polygons = me.polygons
    selected_polygons = [p for p in polygons if p.select]
    
    # extract the face normals - this gives us number of unique faces in something simple like cube
    normal = []
    vertices = []
    for p in selected_polygons:  
        normal.append(p.normal)
        vertices.append(p.vertices)
        
    # find vertices in each mesh and append to a list
    meshes = []
    for i in range(len(vertices)):
        vertices_triangle = [v for v in vertices[i]]
        vertices_triangle.sort()
        meshes.append(vertices_triangle)
    
    # create edges
    background_triangles = []
    background_edges = []
    edges = []
    normals = []
    triangles = []
    edge_pts = []
    background_edge_pts = []
    
    for i in range(len(vertices)):
        for j in range(len(vertices)):
            if i == j:
                continue        
            # count shared points
            count_shared_points = sum(f in meshes[j] for f in meshes[i])
                
            # these are not neighboring triangular meshes
            if count_shared_points < 2:
                continue
            
            # check the surface normals here - if they are the same, then must be on the same face  
            unit_vector_1 = normal[i] / (np.linalg.norm(normal[i]) + 1e-06)
            unit_vector_2 = normal[j] / (np.linalg.norm(normal[j]) + 1e-06)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(dot_product)
           
            if angle < thresh: 
                edge = list(set(meshes[i]) & set(meshes[j]))
                if edge not in background_edges:
                    background_edges.append(edge)
                    background_triangles.append([meshes[i], meshes[j]])
                    background_edge_pts.append(edge[0])
                    background_edge_pts.append(edge[1])
                continue
              
            # find intersection to find the edge
            edge = list(set(meshes[i]) & set(meshes[j]))
         
            if edge not in edges:
                edges.append(edge)
                normals.append([normal[i], normal[j]])
                triangles.append([meshes[i], meshes[j]])
                edge_pts.append(edge[0])
                edge_pts.append(edge[1])
 

    edge_pts = np.unique(np.array(edge_pts))
    background_edge_pts = np.unique(np.array(background_edge_pts))
    
    
    return edge_pts, edges, normals, triangles, background_edge_pts, background_edges, background_triangles

#----------------------------------------------------------------------------------------------------#
# FIND 2D PIXEL VALUES ------------------------------------------------------------------------------#
def pixel_extraction(model, camera_object, me, edge_pts, background_edge_pts, limit=0.05):
    #limit = 0.05
    verts = me.vertices
 
    # render scale and size
    render_scale = bpy.context.scene.render.resolution_percentage / 100
    render_size = (
            int(bpy.context.scene.render.resolution_x * render_scale),
            int(bpy.context.scene.render.resolution_y * render_scale),
            )
    
    output_pixel_coords = []
    coords_dict = {}
    
    # add copies of cubes in correct spots
    for i, co in enumerate(verts):
        
        # make sure it is an edge
        if i not in edge_pts and i not in background_edge_pts:
            continue
        
        co = co.co @ model.matrix_world
        co2D = world_to_camera_view(bpy.context.scene, camera_object, co)
        
         # If inside the camera view
        if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0 and co2D.z >0: 
            bpy.ops.mesh.primitive_cube_add(location=(co))
            bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))
            bpy.data.objects['Cube'].select_set(True)
            
            
            # Try a ray cast, in order to test the vertex visibility from the camera
            location= bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, camera_object.location, (co - camera_object.location).normalized() )
            
            # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
            pixel_coords = (co2D.x * bpy.context.scene.render.resolution_x,
                    co2D.y * bpy.context.scene.render.resolution_y)
            
            if location[0] and (co - location[1]).length < limit:
                pixel_coords = (co2D.x * render_size[0],
                        co2D.y * render_size[1])
                    
                coords_dict[str(i)] = (round(pixel_coords[0]),
                            round(bpy.context.scene.render.resolution_y - pixel_coords[1]))
             
            objs = bpy.data.objects
            objs.remove(objs["Cube"], do_unlink=True)          
            #bpy.ops.object.delete()
                        
    return coords_dict
   

# IS VISIBLE??? ----------------------------------------------------------------------------------#
def is_Visible(model , v,  limit, add_box=False): # v is vector
    
    camera_object = bpy.data.objects["Camera"]
    # render scale and size
    render_scale = bpy.context.scene.render.resolution_percentage / 100
    render_size = (
            int(bpy.context.scene.render.resolution_x * render_scale),
            int(bpy.context.scene.render.resolution_y * render_scale),
            )
            
    
    co = v @ model.matrix_world
    co2D = world_to_camera_view(bpy.context.scene, camera_object, co)
    
        
    # If inside the camera view
    if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0 and co2D.z >0: 
        
        if add_box:
            bpy.ops.mesh.primitive_cube_add(location=(co))
            bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))
            bpy.data.objects['Cube'].select_set(True)
        
        # Try a ray cast, in order to test the vertex visibility from the camera
        location= bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, camera_object.location, (co - camera_object.location).normalized() )
     
        if add_box:
            objs = bpy.data.objects
            objs.remove(objs["Cube"], do_unlink=True) 
            #bpy.ops.object.delete()
                 
        pixel_coords = (co2D.x * render_size[0], co2D.y * render_size[1])
        pix = (round(pixel_coords[0]),
                            round(bpy.context.scene.render.resolution_y - pixel_coords[1]))
        
        if location[0] and (co - location[1]).length < limit:
            return True, pix
         
        else:
            return False, pix

    else:
        return False, None

# helper function for edge pixels
def helper_pixels(model, pt_line, start, pos):
    visible = start
    # add the visible points on line
    for i,l in enumerate(pt_line):
        vis = is_Visible(model, l, limit=0.04, add_box=True)
            
        if vis[0]:
            visible = vis[1]
        else:
            break
        
    difx = abs(start[0] - visible[0])      
    dify = abs(start[1] - visible[1])   
    if difx <= 0 and dify <= 0:
        return []
        
    if pos == 'start':
        return [start, visible]
    elif pos == 'end':
        return [visible, start]
    
        
# ------get pixel values-----------------------------------------------------------------
def edge_pixels(model, me, e0, e1, coords_dict, coords_dict_keys):
    # starting and ending vertice coordinates
    start_vector = me.vertices[e0].co 
    end_vector = me.vertices[e1].co
    N_lo = 8
    N_hi = 50
    
    occlusion = False # set flag for occlusion
    # CASE 1 - if both edge points are visible - no occlusion of edge (sampling N=5 points to check for occlusion)
    if str(e0) in coords_dict_keys and str(e1) in coords_dict_keys:      
        # generate points along the line
        pt_line = points_on_line(start_vector, end_vector, N_lo, endpoint=True)
        
        visible = []
        # add the visible points on line
        for i,l in enumerate(pt_line):
            vis = is_Visible(model, l, limit=0.04, add_box=True)
            
            if vis[0]:
                visible.append(vis[1])
              
        if len(visible) == (N_lo-1):
            start = coords_dict[str(e0)]
            end = coords_dict[str(e1)]
            
            difx = abs(start[0] - end[0])      
            dify = abs(start[1] - end[1])   
            if difx <= 0 and dify <= 0:
                return []
            else:
                return [coords_dict[str(e0)], coords_dict[str(e1)]]
            
        else:
            occlusion = True
            
         
    # CASE 2 - if one edge point is visible - find last visible point   
    elif str(e0) in coords_dict_keys and str(e1) not in coords_dict_keys:
        
        # generate points along line
        pt_line = points_on_line(start_vector, end_vector, N_hi, endpoint=True)
        start = coords_dict[str(e0)]
        return helper_pixels(model, pt_line, start, 'start')
        
    elif str(e0) not in coords_dict_keys and str(e1) in coords_dict_keys:
        # generate points along line
        pt_line = points_on_line(end_vector, start_vector, N_hi, endpoint=True)
        start = coords_dict[str(e1)]
        return helper_pixels(model, pt_line, start, 'end')  
    
    
    # CASE 4- if both edge points are visible - but occlusion occurs within edge
    if str(e0) in coords_dict_keys and str(e1) in coords_dict_keys and occlusion:
        
        return_list = []
        # find from start as beginning of line----
        pt_line = points_on_line(start_vector, end_vector, N_hi, endpoint=True)
        start = coords_dict[str(e0)]
        a =  helper_pixels(model, pt_line, start, 'start')
        if len(a) != 0:
            return_list.append(a[0])
            return_list.append(a[1])
        
        # find from end as beginning of line----
        pt_line = points_on_line(end_vector, start_vector, N_hi, endpoint=True)
        start = coords_dict[str(e1)]
        b =  helper_pixels(model, pt_line, start, 'end')
        if len(b) != 0:
            return_list.append(b[0])
            return_list.append(b[1])

        return return_list
                    
    else:  
        return []



# CLASSIFY if edges should be marked in 2D -----------------------------------------
def background_classification(model, me, background_edges, background_triangles, coords_dict):
    background_edge_dict = {}
    coords_dict_keys = list(coords_dict.keys())
    
    for i, e in enumerate(background_edges):
        e0, e1 = background_edges[i]
        e0, e1 = me.vertices[e0].co, me.vertices[e1].co
        
        v1, v2 = background_triangles[i]
        v1_type = triangle_type(model, me, v1)
        v2_type = triangle_type(model, me, v2)
        
        if v1_type == 'occluding' and v2_type == 'visible':    
            pixels = edge_pixels(model, me, e[0], e[1], coords_dict, coords_dict_keys)
  
        elif v1_type == 'visible' and v2_type == 'occluding':
            pixels = edge_pixels(model, me, e[0], e[1], coords_dict, coords_dict_keys)
   
        else:
            pixels = None
        
        if pixels:
            for j in range(int(len(pixels)/2)):
                    if j == 0:
                        key = str(e)
                    else:
                        key = str(e) + '_' + str(j)
                    
                    background_edge_dict[key] = []
                    start_point = pixels[2*j]
                    end_point = pixels[2*j + 1]
                    background_edge_dict[key].append(start_point)
                    background_edge_dict[key].append(end_point)
                    background_edge_dict[key].append('obscuring') 
 
    
    return background_edge_dict



# CLASSIFY GROUND TRUTHS -------------------------------------------------------------------------#
def contour_classification(model, me, edges, normals, triangles, background_edges, background_triangles, coords_dict):
    
    edge_dict = {}
    coords_dict_values = list(coords_dict.values())
    coords_dict_keys = list(coords_dict.keys())
    
    
    # extract visible edges and normals
    visible_edges = []
    visible_triangles = []
    visible_normals = []
    visible_pixels = []
    
    for i, e in enumerate(edges):
        
        # save edges, triangles, normals
        if str(e[0]) in coords_dict_keys or str(e[1]) in coords_dict_keys:
            pixels = edge_pixels(model, me, e[0], e[1], coords_dict, coords_dict_keys)
            visible_pixels.append(pixels)        
            visible_edges.append(e)
            visible_triangles.append(triangles[i])
            visible_normals.append(normals[i])
    
    # extract points in convex hull
    points = []
    for p in visible_pixels:
        points = points + p

    if len(points) == 0:
        return {}
    
    points = list(dict.fromkeys(points))
    points = np.array(points)
    
    convexHull = cv2.convexHull(points).squeeze()
        
    # step 1: use convex hull to mark certain obscure edges -------------------------
    for i in range(len(visible_edges)):
        if len(visible_pixels[i]) == 0:
            continue
        e = visible_edges[i]
       
        start_point = visible_pixels[i][0]
        end_point = visible_pixels[i][1]
        
        cond = np.all(convexHull == start_point, axis=1)
        idx = np.nonzero(cond)[0].squeeze()
        if idx.size == 0:
            continue         
        elif idx == 0:
            l = len(convexHull)-1
            r = 1
        elif idx == len(convexHull)-1:
            l = len(convexHull)-2
            r = 0
        else: 
            l = idx-1
            r = idx+1
        
        if convexHull[l][0] == end_point[0] and convexHull[l][1] == end_point[1]:
            edge_dict[str(e)] = []
            edge_dict[str(e)].append(start_point)
            edge_dict[str(e)].append(end_point)
            edge_dict[str(e)].append('obscuring')
           
        elif convexHull[r][0] == end_point[0] and convexHull[r][1] == end_point[1]:
            edge_dict[str(e)] = []
            edge_dict[str(e)].append(start_point)
            edge_dict[str(e)].append(end_point)
            edge_dict[str(e)].append('obscuring')

    # step 2: analyze triangles that intersect at edge --------------------------
    for i, e in enumerate(visible_edges):
        if str(e) not in list(edge_dict.keys()):
            v1, v2 = visible_triangles[i]
            v1_type = triangle_type(model, me, v1)
            v2_type = triangle_type(model, me, v2)
            vis_pix = visible_pixels[i]
            
            # step 3 - check occlusion
            if v1_type == 'occluding' or v2_type == 'occluding':
                
                for j in range(int(len(vis_pix)/2)):
                    if j == 0:
                        key = str(e)
                    else:
                        key = str(e) + '_' + str(j)
                    
                    edge_dict[key] = []
                    start_point = vis_pix[2*j]
                    end_point = vis_pix[2*j + 1]
                    edge_dict[key].append(start_point)
                    edge_dict[key].append(end_point)
                    edge_dict[key].append('obscuring') 
       
            # step 4 - concave or convex
            else:
                convexity_type = convexity(model, me, e, visible_triangles[i], visible_normals[i])
                if convexity_type == None:
                    continue
                for j in range(int(len(vis_pix)/2)):
                    if j == 0:
                        key = str(e)
                    else:
                        key = str(e) + '_' + str(j)
                    
                    edge_dict[key] = []
                    start_point = vis_pix[2*j]
                    end_point = vis_pix[2*j + 1]
                    edge_dict[key].append(start_point)
                    edge_dict[key].append(end_point)
                    edge_dict[key].append(convexity_type)   
                 
    return edge_dict