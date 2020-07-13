# maze.py
# goal: draw a maze that looks like a picture
# author: Vilas Winstein

################################################################
# FUNCTIONS FOR GENERATING A SET OF VERTICES BASED ON AN IMAGE #
################################################################

import imageio
import numpy as np
import random

# import an image from a filename
# returns an image matrix
def import_image(filename, channel='brightness'):
    pic = imageio.imread(filename)

    if len(pic.shape) == 2:
        return pic

    if (channel == 'brightness'):
        return np.array([ [ int(0.2126*x[0] + 0.7152*x[1] + 0.0722*x[2]) for x in row ] for row in pic ])

    if (channel == 'red'):
        return pic[:,:,0]

    if (channel == 'green'):
        return pic[:,:,1]

    if (channel == 'blue'):
        return pic[:,:,2]

    raise ValueError("Value of string parameter 'channel' must be 'brightness', 'red', 'green', or 'blue'")

# take an image and turn it into a matrix of probabilities.
# resolution is a tuple, and the image is a matrix of 0-255 values.
# in the original image, lower values are darker, but in the matrix of probabilities, lower values are lighter.
def probabilities_from_image(image, resolution=None, invert=False):
    rowskip = 1
    colskip = 1
    numrows = image.shape[0]
    numcols = image.shape[1]

    if resolution != None:
        rowskip = int(image.shape[0] / resolution[0])
        colskip = int(image.shape[1] / resolution[1])
        numrows = resolution[0]
        numcols = resolution[1]

    probabilities = None

    if invert:
        probabilities = np.array([[ image[(i * rowskip, j * colskip)] for j in range(numcols) ] for i in range(numrows) ])
    else:
        probabilities = np.array([[ 255 - image[(i * rowskip, j * colskip)] for j in range(numcols) ] for i in range(numrows) ])
    
    for row in range(probabilities.shape[0]):
        probabilities[(row,0)] = 255
        probabilities[(row,numcols-1)] = 255

    for col in range(probabilities.shape[1]):
        probabilities[(0,col)] = 255
        probabilities[(numrows-1,col)] = 255

    return probabilities

# take a matrix of probabilities and return the indicator matrix of a random vertex set.
# the random vertex set will be a matrix with the same dimension as the input, containing a 1 if the vertex is in the set and a 0 otherwise.
def random_vertex_indicator_matrix(probabilities):
    return np.array([[random.choices([0,1],weights=[255-x,x])[0] for x in row] for row in probabilities])


#########################################################################
# FUNCTIONS FOR MAKING AN ADJACENCY MATRIX OUT OF VERTICES IN THE PLANE #
#########################################################################

import math

# extract the indices of the random vertex set that contain a 1
def get_vertices(indicatormatrix):
    vertices = []

    for row in range(indicatormatrix.shape[0]):
        for col in range(indicatormatrix.shape[1]):
            if indicatormatrix[(row,col)] == 1:
                vertices += [(row,col)]

    random.shuffle(vertices)
    return vertices

# get euclidean distance between two points
def euclidean_distance(p, q):
    return math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

# get l1 distance between two points
def l1_distance(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

# see if two vertices are adjacent and on the boundary
def adjacent_boundary_vertices(p, q, imageshape):
    if l1_distance(p,q) == 1:
        if p[0] == 0 and q[0] == 0:
            return True
        if p[1] == 0 and q[1] == 0:
            return True
        if p[0] == imageshape[0] - 1 and q[0] == imageshape[0] - 1:
            return True
        if p[1] == imageshape[1] - 1 and q[1] == imageshape[1] - 1:
            return True

    return False

# see if the given pair of vertices are the boundary of the entry door
def is_door(p, q, imageshape):
    if set([p,q]) == set([(0,0), (1,0)]):
        return True

# create a weighted adjacency matrix for the vertices
# the boundary edges will all have minimum weight (of -1), except for the entry door
# WARNING: Nonzero randomness may result in an unsolvable maze.
# However, zero randomness does make the maze look homogeneous near the walls.
def adjacency_matrix(vertices, imageshape, l1=False, randomness=0.1):
    mat = None
    distance = None
    maxnum = None
    n = len(vertices)
    mat = np.zeros((n,n))
    maxnum = float('inf')

    if l1:
        distance = l1_distance
    else:
        distance = euclidean_distance

    for i in range(n):
        for j in range(n):
            if i == j:
                mat[(i,j)] = maxnum
            elif adjacent_boundary_vertices(vertices[i], vertices[j], imageshape):
                if is_door(vertices[i], vertices[j], imageshape):
                    mat[(i,j)] = maxnum
                else:
                    mat[(i,j)] = 0
            else:
                mat[(i,j)] = random.gauss(distance(vertices[i], vertices[j]), randomness)

    return mat


####################################################
# PRIM'S ALGORITHM (COPIED FROM GEEKSFORGEEKS.ORG) #
####################################################

# A utility function to find the vertex with
# minimum distance value, from the set of vertices
# not yet included in shortest path tree
def minKey(key, mstSet):
    # Initilaize min value
    min_value = float('inf')
    
    for v in range(len(key)):
        if key[v] < min_value and mstSet[v] == False:
            min_value = key[v]
            min_index = v

    return min_index

# Function to construct and return MST for a graph  
# represented using adjacency matrix representation 
def primMST(graph): 
    maxnum = float('inf')

    n = graph.shape[0]
  
    # Key values used to pick minimum weight edge in cut 
    key = [maxnum] * n
    parent = [None] * n # Array to store constructed MST 
    # Make key 0 so that this vertex is picked as first vertex 
    key[0] = 0 
    mstSet = [False] * n

    parent[0] = -1 # First node is always the root of the MST
  
    for cout in range(n): 

        # Pick the minimum distance vertex from  
        # the set of vertices not yet processed.  
        # u is always equal to src in first iteration 
        u = minKey(key, mstSet) 
  
        # Put the minimum distance vertex in  
        # the shortest path tree 
        mstSet[u] = True
  
        # Update dist value of the adjacent vertices  
        # of the picked vertex only if the current  
        # distance is greater than new distance and 
        # the vertex in not in the shotest path tree 
        for v in range(n): 
  
            # graph[u][v] is non zero only for adjacent vertices of m 
            # mstSet[v] is false for vertices not yet included in MST 
            # Update the key only if graph[u][v] is smaller than key[v] 
            if mstSet[v] == False and key[v] > graph[(u,v)]: 
                    key[v] = graph[(u,v)] 
                    parent[v] = u 
  
    return parent


######################################################################
# FUNCTIONS FOR TURNING THE OUTPUT OF primMST INTO A DRAWABLE FORMAT #
######################################################################

# turns the 'parent' list given by primMST into a list of edges
# the list of edges is a list of lists, each containing two points
def tree_from_parent(parent, vertices):
    tree = []

    for i in range(1, len(parent)):
        tree = tree + [[vertices[i], vertices[parent[i]]]]

    return tree

# cuts a hole in the tree so the maze is solvable
def remove_other_door(tree, imageshape):
    try:
        tree.remove([(imageshape[0]-2, imageshape[1]-1), (imageshape[0]-1, imageshape[1]-1)])
    except ValueError:
        tree.remove([(imageshape[0]-1, imageshape[1]-1), (imageshape[0]-2, imageshape[1]-1)])
    except:
        print('Something else went wrong in remove_other_door')


##################################
# FUNCTIONS FOR DRAWING THE MAZE #
##################################

import graphics
from PIL import Image as NewImage

# turn a tuple into a point on the plane
def translate_point(p, borderwidth, wallwidth, hallwaywidth):
    x_val = borderwidth + wallwidth * (p[1] + 1) + hallwaywidth * p[1] - int(wallwidth/2)
    y_val = borderwidth + wallwidth * (p[0] + 1) + hallwaywidth * p[0] - int(wallwidth/2)
    return graphics.Point(x_val, y_val)

# draw a graph. vertices is a list of points, edges is a list of pairs of points.
# NOTE: savefilename should not include a .gif extension. That extension will be added automatically.
def draw_graph(vertices, edges, title='maze', wallcolor='black', bgcolor='white', wallwidth=5, hallwaywidth=8, borderwidth=20, savefilename=None):
    max_x_value = max([ v[1] for v in vertices])
    max_y_value = max([ v[0] for v in vertices])

    totalwidth  = 2 * borderwidth + wallwidth * (max_x_value + 1) + hallwaywidth * max_x_value
    totalheight = 2 * borderwidth + wallwidth * (max_y_value + 1) + hallwaywidth * max_y_value
    
    window = graphics.GraphWin(title, totalwidth, totalheight)
    window.setBackground(bgcolor)

    for v in vertices:
        pt = graphics.Circle(translate_point(v, borderwidth, wallwidth, hallwaywidth), int(wallwidth/2))
        pt.setFill(wallcolor)
        pt.draw(window)

    for e in edges:
        p0 = translate_point(e[0], borderwidth, wallwidth, hallwaywidth)
        p1 = translate_point(e[1], borderwidth, wallwidth, hallwaywidth)
        ln = graphics.Line(p0,p1)
        ln.setFill(wallcolor)
        ln.setWidth(wallwidth)
        ln.draw(window)

    if savefilename != None:
        window.postscript(file=savefilename+'.eps', colormode='color')
        img = imageio.imread(savefilename+'.eps')
        imageio.imwrite(savefilename+'.png', img)

    window.getMouse()
    window.close()


########################################
# WHEN THIS FILE IS RUN AS A SCRIPT... #
########################################

import sys
import ast
import time

# time the program in splits
def timesplit(title, titlewidth=30, decimalplaces=5):
    global last_checkpoint
    this_checkpoint = time.time()

    if verbose:
        print(title, 'took', ' '*(titlewidth-len(title)), ('{:.'+str(decimalplaces)+'f}').format(this_checkpoint - last_checkpoint), 'seconds')

    last_checkpoint = this_checkpoint

def is_pair_of_positive_ints(a):
    if type(a) != list:
        return False
    if len(a) != 2:
        return False
    if type(a[0]) != int or type(a[1]) != int:
        return False
    if a[0] <= 0 or a[1] <= 0:
        return False

    return True

def is_nonnegative_real(a):
    if type(a) != float and type(a) != int:
        return False
    if a < 0:
        return False

    return True

def is_positive_int(a):
    if type(a) != int:
        return False
    if a <= 0:
        return False

    return True

def print_help_message():
    print('maze.py is a program that creates a maze that looks like any picture.')
    print('WARNING: this program is very slow!')
    print('Making a maze that from an n x m image, will take O(n^2 m^2) time!')
    print('Here are some examples of typical usage:')
    print()
    print('\tpython maze.py lenna.png -resolution [64,64] -out lenna_maze')
    print('\tpython maze.py -v pictures/lenna.png -channel blue -wallcolor blue')
    print()
    print()
    print('Boolean Options:')
    print()
    print('\t-v            verbose: prints runtimes for each part of the program.')
    print()
    print('\t-invert       inverts the brightness values of the image, so dark')
    print('\t              parts become light and vice versa.')
    print()
    print('\t-l1           makes the program use the l1 distance instead of the')
    print('\t              euclidean (l2) distance when generating the minimum')
    print('\t              spanning tree. This means that the maze will look')
    print('\t              more grid-like (I think... I\'m not sure though).')
    print()
    print()
    print('String Options:')
    print()
    print('\t-in           denotes the input filename. By default this argument')
    print('\t              is not necessary, and the first string in the list of')
    print('\t              arguments that is not part of any other argument will')
    print('\t              be used as the input filename.')
    print()
    print('\t-out          denotes the output filename. As with -in, by default')
    print('\t              this argument is not necessary, and the second free')
    print('\t              string in the list of arguments will be taken as the')
    print('\t              output filename.')
    print('\t              NOTE that the argument to -out should not include an')
    print('\t              extension (like .png), since if output is requested,')
    print('\t              it will be generated to two different files, an .eps')
    print('\t              file and a .png file, with the same name before the')
    print('\t              extension.')
    print('\t              ALSO NOTE that if no output filename is specified,')
    print('\t              the program will not produce an output file, and')
    print('\t              will instead only display the maze on screen.')
    print()
    print('\t-channel      denotes the source channel for the brightness values')
    print('\t              used by the program.')
    print('\t              Options are: brightness, red, blue, or green.')
    print('\t              Default value is \'-channel brightness\'.')
    print()
    print('\t-wallcolor    is the color for drawing the walls of the maze.')
    print('\t              Default value is \'-wallcolor black\'.')
    print()
    print('\t-bgcolor      is the color for drawing the background of the maze.')
    print('\t              Default value is \'-bgcolor white\'.')
    print()
    print()
    print('Integer Options:')
    print()
    print('\t-wallwidth    is the width of the maze walls when drawn.')
    print('\t              Default value is \'-wallwidth 5\'.')
    print()
    print('\t-hallwaywidth is the width of the maze hallways when drawn.')
    print('\t              Default value is \'-hallwaywidth 8\'.')
    print()
    print('\t-borderwidth  is the width of the maze border when drawn.')
    print('\t              Default value is \'-borderwidth 20\'.')
    print()
    print()
    print('Other Options:')
    print()
    print('\t-resolution   is the resolution of the maze, meaning that if the')
    print('\t              maze is completely full of maze (i.e. if the image')
    print('\t              was completely black), and if the resolution is [n,m]')
    print('\t              then the maze will have n-1 vertical columns and m-1')
    print('\t              horizontal rows.')
    print('\t              WARNING: this is the part that scales poorly, i.e.')
    print('\t              if -resolution [n,m] is passed, then the program will')
    print('\t              run in O(n^2 m^2) time. And if the -resolution')
    print('\t              argument is not included, then the resolution of the')
    print('\t              maze will default to the resolution of the input image')
    print('\t              which could be very large!')
    print()
    print('\t-randomness   is an argument that makes the maze look a little more')
    print('\t              random and natural. If -randomness 0 is passed, then')
    print('\t              the output maze will look very homogeneous along the')
    print('\t              edges.')
    print('\t              WARNING: if the randomness is set too high, then it is')
    print('\t              possible for an unsolvable maze to be generated!')
    print('\t              Default value is -randomness 0.1, which in practice')
    print('\t              has not created any unsolvable mazes yet.')
    print()
    print('\t-help         prints this dialog.')
    print()

boolean_args = ['-l1', '-invert', '-v']
string_args  = ['-in', '-out', '-channel', '-wallcolor', '-bgcolor']
int_args     = ['-wallwidth', '-hallwaywidth', '-borderwidth']
other_args   = {
        '-resolution' : (is_pair_of_positive_ints, 'a pair of positive integers'),
        '-randomness' : (is_nonnegative_real, 'a nonnegative real number')
    }


# MAIN CODE BLOCK
if __name__ == '__main__':

    # DEFAULT ARGUMENTs

    arg_dict = {
            '-l1' : False,
            '-invert' : False,
            '-v' : False,
            
            '-in' : None,
            '-out' : None,
            '-channel' : 'brightness',
            '-wallcolor' : 'black',
            '-bgcolor' : 'white',
            
            '-wallwidth' : 5,
            '-hallwaywidth' : 8,
            '-borderwidth' : 20,
            
            '-resolution' : None,
            '-randomness' : 0.1
        }


    # PARSE USER ARGUMENTS

    arguments = sys.argv[1:]

    arguments_lower = [ x.lower() for x in arguments ]

    if '-help' in arguments_lower or '--help' in arguments_lower or 'help' in arguments_lower:
        print_help_message()
        sys.exit()

    for arg in boolean_args:
        if arg in arguments:
            index = arguments.index(arg)
            del arguments[index]

            arg_dict[arg] = True

    for arg in string_args:
        if arg in arguments:
            index = arguments.index(arg)
            del arguments[index]

            if index >= len(arguments):
                print('Error: no argument given for', arg)
                sys.exit()

            arg_dict[arg] = arguments[index]
            del arguments[index]

    for arg in int_args:
        if arg in arguments:
            index = arguments.index(arg)
            del arguments[index]

            if index >= len(arguments):
                print('Error: no argument given for', arg)
                sys.exit()

            try:
                arg_dict[arg] = ast.literal_eval(arguments[index])
            except:
                print('Error: couldn\'t parse argument to', arg)
                sys.exit()

            del arguments[index]

            if not is_positive_int(arg_dict[arg]):
                print('Error:', arg, 'argument,', arg_dict[arg], 'is not a positive integer')
                sys.exit()
                
    for arg in other_args:
        if arg in arguments:
            index = arguments.index(arg)
            del arguments[index]

            if index >= len(arguments):
                print('Error: no argument given for', arg)
                sys.exit()

            try:
                arg_dict[arg] = ast.literal_eval(arguments[index])
            except:
                print('Error: couldn\'t parse argument to', arg)
                sys.exit()

            del arguments[index]

            if not other_args[arg][0](arg_dict[arg]):
                print('Error:', arg, 'argument,', arg_dict[arg], ', is not', other_args[arg][1])
                sys.exit()

    if arg_dict['-in'] == None:
        if len(arguments) < 1:
            print('Error: no input filename given')
            sys.exit()
        else:
            arg_dict['-in'] = arguments[0]
            del arguments[0]

    if arg_dict['-out'] == None and len(arguments) > 0:
        arg_dict['-out'] = arguments[0]
        del arguments[0]

    if len(arguments) > 0:
        print('Error: extraneous arguments given.\nType \'python maze.py -help\' for help.')
        sys.exit()

    if arg_dict['-resolution'] != None:
        arg_dict['-resolution'] = (arg_dict['-resolution'][1], arg_dict['-resolution'][0])


    # ACTUALLY RUN THE PROGRAM

    verbose = arg_dict['-v']

    last_checkpoint = time.time()

    image = import_image(arg_dict['-in'], channel=arg_dict['-channel'])
    timesplit('import_image')

    probs = probabilities_from_image(image, resolution=arg_dict['-resolution'], invert=arg_dict['-invert'])
    timesplit('probabilities_from_image')

    if arg_dict['-resolution'] == None:
        imageshape = probs.shape
    else:
        imageshape = arg_dict['-resolution']

    vim = random_vertex_indicator_matrix(probs)
    timesplit('random_vertex_indicator_matrix')

    vertices = get_vertices(vim)
    timesplit('get_vertices')

    graph = adjacency_matrix(vertices, imageshape, l1=arg_dict['-l1'], randomness=arg_dict['-randomness'])
    timesplit('adjacency_matrix')
    
    parent = primMST(graph)
    timesplit('primMST')

    tree = tree_from_parent(parent, vertices)
    timesplit('tree_from_parent')

    remove_other_door(tree, imageshape)
    timesplit('remove_other_door')

    draw_graph(vertices, tree, title=('maze generated from ' + arg_dict['-in']), wallcolor=arg_dict['-wallcolor'], bgcolor=arg_dict['-bgcolor'], wallwidth=arg_dict['-wallwidth'], hallwaywidth=arg_dict['-hallwaywidth'], borderwidth=arg_dict['-borderwidth'], savefilename=arg_dict['-out'])
