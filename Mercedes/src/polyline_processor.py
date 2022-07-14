import shapefile
import networkx as nx
import numpy as np
from pyflann import *
import shp_parser
from shapely.geometry import Point, LineString
import horizon
import matplotlib.pyplot as plt


case = 24
nPoints = 1508
shpFileName = f"../../lane_select/case_{case}_lanecenter.shp"
trajFileName = f"../../amap/trajectory/case_{case}_fake_trajectory_{nPoints}_final.shp"
ptsSavePath = f"../../gen_trajectory/case_{case}_points_new.shp"
pLineSavePath = f"../../gen_trajectory/case_{case}_pLine_new.shp"
txtSavePath = f"../../txt/case_{case}_points_new.txt"


ERROR = 1e-6    # Maximum Permissible Error


def generate_graph(shapes):
    """
    Operation object: shapefile that has a `ShapeType` of 3 (PolyLine)

    Input `shapes()` method of a shapefile and return a `Graph`
    that contains each vertex of the PolyLine. Every edge of the
    graph is defined by shapefile geometry.

    By judging the number of connnected subgraph to see if we 
    have a engineering error. If so, call `connect_subgraph()`
    to connect the two points. 
    """
    G = nx.Graph()
    for i in range(len(shapes)):
        for j in range(len(shapes[i].points) - 1):
            G.add_edge(shapes[i].points[j], shapes[i].points[j + 1])
            # print(shapes[i].points[j], shapes[i].points[j + 1])
    nSubgraph = nx.number_connected_components(G)
    if nSubgraph > 1:
        print(f"Number of Subgraph = {nSubgraph}")
        G = connect_subgraph(G)     
    return G

def get_start_point(shapes):
    # FIXME
    idxStart = shapes[0].parts[0]
    start = shapes[0].points[idxStart]
    return start


def connect_subgraph(G):
    """
    When the distance between two points is less than `ERROR`
    predefined, we consider these two to be the same point.

    Use `flann.nn()` method to find the nearest neighbour of 
    each point. Then connect the two points in the Graph.
    """
    graphList = list(G)
    dataset = testset = np.array([[p[0], p[1]] for p in graphList])
    flann = FLANN()
    result, distance = flann.nn(dataset, testset, 2, algorithm="kmeans", 
                                branching=32, iterations=7, checks=16) 
    del flann
    # XXX
    for idx, dist in enumerate(distance):
        if dist[1] < ERROR:
            G.add_edge(graphList[result[idx][0]], graphList[result[idx][1]])

    return G

def eulerpath_to_list(G, reverse=False):
    """Return a list of points on Eulerian path in sequence
    
    Find an Euler path from the Graph, if the Graph doesn't have
    an eulerian path, replace it with its `largest connected subgraph`
    and find its Euler path.

    On the Euler path, points are arranged in order. A new list 
    with no duplicate points is to be returned.
    
    Parameters
    ----------
    G: NetworkX graph 
        The graph that you want to find an Eulerian path from.

    reverse: a flag 
        To define if you want the path reversed. If you do, let
        `reverse=True`.

    """
    if nx.has_eulerian_path(G) == False:
        print("`G` doesn't have an Eulerian path")
        # XXX
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest)
    graphList = list(nx.eulerian_path(G))
    if reverse == True:
        graphList = list(nx.eulerian_path(G, graphList[-1][-1]))

    tmpList = []
    for i in range(len(graphList)):
        for j in range(2):
            tmpList.append(graphList[i][j])  
    newList = list(set(tmpList))
    newList.sort(key=tmpList.index)
    del tmpList
    return newList

def get_points_from_trajectory(fName):
    pts = []
    sf = shapefile.Reader(fName)
    shps = sf.shapes()
    len = sf.numShapes
    for idx in range(len):
        pts.append(Point(shps[idx].points[0]))
    return pts

def polyline_interpolation_piecewise(pLine, ptsList):
    """Return a list of interpolated points.

    Perform piecewise linear interpolation on pLine
   
    Parameters
    ----------
    pLine: A `LineString` type polyline
    
    ptsList: A list of points
        The returned `ptsIntplt` has the same length with
        `ptsList`
    
    """
    ptsIntplt = []
    n = len(ptsList)
    for i in range(n):
        dist = float(i) / n
        ptTmp = pLine.interpolate(dist, normalized=True)
        ptsIntplt.append((ptTmp.x, ptTmp.y))
    return ptsIntplt

def polyline_interpolation_byPoints(pLine, ptsList):
    """Return a list of interpolated points.

    Project every single point of trajectory to pLine.

    Parameters
    ----------
    pLine: A `LineString` type polyline
    
    ptsList: A list of points
        you want to project those points onto polyline.

    """
    ptsIntplt = []
    for pt in ptsList:
        ptTmp = pLine.interpolate(pLine.project(pt))
        ptsIntplt.append((ptTmp.x, ptTmp.y))
    return ptsIntplt

def lineString_to_list(pLine):
    x, y = pLine.xy
    xList, yList = list(x), list(y)
    length = len(xList)
    newList = []
    for i in range(length):
        newList.append([xList[i], yList[i]])
    return newList

def polyline_plot(pLine):
    x, y = pLine.xy
    xList, yList = list(x), list(y)
    plt.plot(xList, yList)
    plt.show()

def list_to_txt(fName, list):
    f = open(fName, 'w')
    for element in list:
        f.write(f"{element}\n")
    f.close()

def main():
    sf = shapefile.Reader(shpFileName)
    shps = sf.shapes()
    line = eulerpath_to_list(generate_graph(shps), reverse=True)
    # print(line)
    pLine = LineString(line)
    pts = get_points_from_trajectory(trajFileName)
    ptsIntplt = polyline_interpolation_piecewise(pLine, pts)
    shp_parser.savePtsSHP(ptsSavePath, ptsIntplt)
    shp_parser.saveLineSHP(pLineSavePath, ptsIntplt)
    
    pLineNew, pLength = horizon.pdis2(ptsIntplt)
    list_to_txt(txtSavePath, lineString_to_list(pLineNew))
    polyline_plot(pLineNew)
    print(f"length of polyline: {pLength}")

    
if __name__ == "__main__":
    main()

