import pickle
import numpy as np
import time
import math 

def distance(point1, point2):
    '''
    input: two points (in 3-d) -> format [(x,y,z),<binary flag for cloud>]
    output: distance
    function: Uses distance formula if points are from different clouds, else returns infinity
    '''
    if point1[1] == point2[1]:
        return float('inf')
    return ((point1[0][0] - point2[0][0]) ** 2 + (point1[0][1] - point2[0][1]) ** 2 + (point1[0][2] - point2[0][2]) ** 2) ** 0.5

def closest_pair(points):
    '''
    
    '''
    n = len(points)
    
    if n <= 25:
        #If the number  of points is small, use brute force to find the closest pair
        min_dist = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                if points[i][1]!= points[j][1]:
                    min_dist = min(min_dist, distance(points[i], points[j]))
        return min_dist
    
    # Sort the points by x-coordinate
    sorted_points = sorted(points, key=lambda point: point[0][0])
    
    # Divide the points into two halves
    mid = n // 2
    left_half = sorted_points[:mid]
    right_half = sorted_points[mid:]
    
    # Recursively find the minimum distance in each half
    min_dist_left = closest_pair(left_half)
    min_dist_right = closest_pair(right_half)
    
    # Get the minimum of the two minimum distances
    min_dist = min(min_dist_left, min_dist_right)
    
    # Merge the two halves and find the minimum distance among the pairs with one point in each half
    strip = []
    mid_x = (left_half[-1][0][0] + right_half[0][0][0]) / 2

    if min_dist != float('inf'):
        for point in sorted_points:
            if abs(point[0][0] - mid_x) < min_dist:
                strip.append(point)
        # print(len(strip))
        strip_min = min(distance(strip[i], strip[j]) for i in range(min(15,len(strip))) for j in range(i + 1, 15))

        return min(min_dist, strip_min)
    
    else:
        # Sweep the plane in the left direction and find a point on the left side
        left_point = None
        for point in reversed(left_half):
            left_point = point
            break

        # Sweep the plane in the right direction and find a point on the right side
        right_point = None
        for point in right_half:
            right_point = point
            break

        # Calculate the distance between the left and right points
        min_dist = distance(left_point, right_point)
        return min_dist
        
def add_flag(pointsA, pointsB):
    '''
    input: two 3-d points
    output: points with flag
    '''
    points = [(x, 0) for x in pointsA] + [(y, 1) for y in pointsB]
    return points

# Script 1

# this file contains the points from over 1600 different components (each component is a cloyd)
with open('filename2.pickle', "rb") as input_file:
    output= pickle.load(input_file)

faces = output[1]
vertices = output[0].squeeze()

import scipy.sparse
# make sparse matrix
nonzero_values = np.ones(faces.shape[0]*2)
row_indices = np.hstack([faces[:,0],faces[:,0]])
column_indices = np.hstack([faces[:,1],faces[:,2]])

num_rows = max(row_indices) + 1
num_columns = max(column_indices) + 1

sparse_matrix = scipy.sparse.coo_matrix((nonzero_values, (row_indices, column_indices)), shape=(num_rows, num_columns))
# find connected components
n_components, labels = scipy.sparse.csgraph.connected_components(sparse_matrix, directed=False)

# print(n_components)
# # for each pair of connected components, find minimum distance
store = []
for idx in range(n_components):
    store.append(np.where(labels==idx)[0])
#    if len(store[-1]) > 10_000:
#        store[-1] = store[-1][::100]

start_time_script1 = time.time()
cc_dists1 = np.zeros([n_components,n_components]) 
for idx1 in range(n_components):
    for idx2 in range(idx1):
        # print(len(vertices[store[idx1],:]),len(vertices[store[idx2],:]))
        if len(vertices[store[idx1],:]) > 400 and len(vertices[store[idx2],:]) > 400:
            #in cases less than this, the naive approach works better.
            print((idx1,idx2) + (len(vertices[store[idx1],:]),len(vertices[store[idx2],:])))
            points = add_flag(vertices[store[idx1],:],vertices[store[idx2],:])
            sorted_points = sorted(points, key=lambda point: point[0][0])
            cc_dists1[idx1,idx2] = closest_pair(sorted_points)
        else:
            cc_dists1[idx1,idx2] = np.min(scipy.spatial.distance.cdist(vertices[store[idx1],:],
                                                                vertices[store[idx2],:]))
        
print(cc_dists1)                 
end_time_script1 = time.time()
execution_time_script1 = end_time_script1 - start_time_script1
print(f"Script 1 Execution Time: {execution_time_script1} seconds")



# Script2
# Runs naive approach to check run time and correctness of algorithms

with open('filename2.pickle', "rb") as input_file:
    output= pickle.load(input_file)

faces = output[1]
vertices = output[0].squeeze()

import scipy.sparse
# make sparse matrix
nonzero_values = np.ones(faces.shape[0]*2)
row_indices = np.hstack([faces[:,0],faces[:,0]])
column_indices = np.hstack([faces[:,1],faces[:,2]])

num_rows = max(row_indices) + 1
num_columns = max(column_indices) + 1

sparse_matrix = scipy.sparse.coo_matrix((nonzero_values, (row_indices, column_indices)), shape=(num_rows, num_columns))
# find connected components
n_components, labels = scipy.sparse.csgraph.connected_components(sparse_matrix, directed=False)

# print(n_components)
# # for each pair of connected components, find minimum distance
store = []

for idx in range(n_components):
    store.append(np.where(labels==idx)[0])
#    if len(store[-1]) > 10_000:
#        store[-1] = store[-1][::100]

start_time_script2 = time.time()
cc_dists = np.zeros([n_components,n_components]) 
for idx1 in range(n_components):
    # print(idx1)
    for idx2 in range(idx1):
        cc_dists[idx1,idx2] = np.min(scipy.spatial.distance.cdist(vertices[store[idx1],:],
                                                                vertices[store[idx2],:]))
        
print(cc_dists)                 
end_time_script2 = time.time()
execution_time_script2 = end_time_script2 - start_time_script2
print(f"Script 2 Execution Time: {execution_time_script2} seconds")

# Checks correctness
print(cc_dists1==cc_dists)