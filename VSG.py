# script file:
# -------------------------------- lab data---------------------
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import matplotlib.path as mplPath

def wrap_angle(angle):
    """ Wraps angle between -pi and pi

    @type  angle: float or numpy array
    @param angle: angle in radinas

    @rtype:   float or numpy array
    @return:  angle in radinas between -Pi to Pi
    """
    if isinstance(angle, float) or isinstance(angle, int):
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    elif isinstance(angle, np.ndarray):
        return (angle + np.pi) % (2 * np.pi ) - np.pi
    elif isinstance(angle, list):
        ret = []
        for i in angle:
            ret.append(wrap_angle(i))
        return ret
    else:
        raise NameError('wrap_angle')

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.alph = 0

    def alpha(self, x):
        self.alph = x

    def dist(self, p):
        return np.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)

    def numpy(self):
        return np.array([self.x, self.y])

    def dist_line(self, l):
        return np.linalg.norm(np.cross(l.p2.numpy() - l.p1.numpy(), l.p1.numpy() - self.numpy())) / np.linalg.norm(l.p2.numpy() - l.p1.numpy())

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y,self.alph)

    def dot(self, p):
        return self.x * p.x + self.y*p.y

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def vector(self, p):
        return Point(p.x - self.x, p.y - self.y)

    def unit(self):
        mag = self.length()
        return Point(self.x/mag, self.y/mag)

    def scale(self, sc):
        return Point(self.x * sc, self.y * sc)

    def add(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __truediv__(self, s):
        return Point(self.x / s, self.y / s)

    def __floordiv__(self, s):
        return Point(int(self.x / s), int(self.y / s))

    def __mul__(self, s):
        return Point(self.x * s, self.y * s)

    def __rmul__(self, s):
        return self.__mul__(s)

    def dist_segment(self, s):
        line_vec = s.p1.vector(s.p2)
        pnt_vec = s.p1.vector(self)
        line_len = line_vec.length()
        line_unitvec = line_vec.unit()
        pnt_vec_scaled = pnt_vec.scale(1.0/line_len)
        t = line_unitvec.dot(pnt_vec_scaled)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        nearest = line_vec.scale(t)
        dist = nearest.dist(pnt_vec)
        nearest = nearest.add(s.p1)
        return dist

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y,self.alph)

def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) >= (B.y - A.y) * (C.x - A.x)

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

class Segment:
    def __init__(self, p1=Point(), p2=Point()):
        self.p1 = p1
        self.p2 = p2
        self.distance = 0

    @classmethod
    def point_angle_length(cls, p1=Point(), angle=0, length=1):
        x2 = p1.x + math.cos(angle) * length
        y2 = p1.y + math.sin(angle) * length
        return cls(p1, Point(x2, y2))

    # Return true if line segments AB and CD intersect
    def intersect(self, s):
        if ccw(self.p1, s.p1, s.p2) != ccw(self.p2, s.p1, s.p2) and ccw(self.p1, self.p2, s.p1) != ccw(self.p1, self.p2, s.p2):
            return True, self.intersection_point(s)
        else:
            return False, None

    def intersection_point(self, line):
        xdiff = (self.p1.x - self.p2.x, line.p1.x - line.p2.x)
        ydiff = (self.p1.y - self.p2.y, line.p1.y - line.p2.y)

        div = det(xdiff, ydiff)
        if div == 0:
            #print("Something went wrong!")
            return None

        d = (det((self.p1.x, self.p1.y), (self.p2.x, self.p2.y)), det((line.p1.x, line.p1.y), (line.p2.x, line.p2.y)))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return Point(x, y)

    def dis(self, x):
        self.distance = x

    def magnitude(self):

        return math.sqrt(((self.p2.x-self.p1.x)**2)+(self.p2.y-self.p1.y)**2)

    def angle(self):

        return abs(math.atan2((self.p2.x-self.p1.x),(self.p2.y-self.p1.y)))


    def __str__(self):
        return "[{}, {}]".format(self.p1, self.p2)

# ----------------------------------------------------------------------
# -------------------------------- VERTEXES + EDGES  ---------------------

class Visibility:
    def __init__(self, map):
        self.map = map

 # This function reads a csv file to retrieve the vertexes in the environment (start, goal, and obstacles' vertexes)
 # It creates a dictionary of lists where each list represents the vertexes in a specific obstacle

    def get_edges_vertexes(self):

      # vertexes from csv
        v = []
        with open(self.map, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)
            for row in csv_reader:
                v.append(row)

        # list -> arr
        v = np.array(v)
        v = v.astype(float)

        # arr -> dict
        vertexes_dict = {}

        # obstacle identifier, int(i[0])
        # value =[]
        # adds curr vertex (Point) for the corresponding obstacle (using obstacle indentifier) in the vertexes_dict

        for i in v:
            if not (bool(vertexes_dict.get(int(i[0])))):
                vertexes_dict[int(i[0])] = []
            vertexes_dict[int(i[0])].append(Point(i[1], i[2]))

        # This function also computes the Edges of each obstacle considering the vertexes that define it

        E = []

        # loop = obstacles except first(start) and last (goal)
        for i in range(1, len(vertexes_dict) - 1):

          # loop = each obstacle's vertex
            for j in range(0, len(vertexes_dict[i]) - 1):

              # connect= curr vertex -> next vertex
                E.append(Segment(vertexes_dict[i][j], vertexes_dict[i][j + 1]))

                # connect last vertex to the first vertex of the obstacle
            E.append(Segment(vertexes_dict[i][len(vertexes_dict[i]) - 1], vertexes_dict[i][0]))

        return E, vertexes_dict

# plot obstacle edges
    def plot_edges(self, edges):
        for j in edges:
            plt.plot([j.p1.x, j.p2.x], [j.p1.y, j.p2.y], color='black', linewidth=3)

# plot vertexes
    def plot_vertexes(self, vertexes):

      # for how many vertexes plotted?
        counter = 0

        # i = index
        # j = vertex

        for i in range(0, len(vertexes)):
            for j in vertexes[i]:

                plt.scatter(j.x, j.y, color='green', zorder=3)
                plt.annotate(counter, (j.x + 0.1, j.y))
                counter += 1

# visibility edges emanating from each vertex = potential edges
    def plot_visibility(self, visibility_edges):
        for j in visibility_edges:
            plt.plot([j.p1.x, j.p2.x], [j.p1.y, j.p2.y], color='blue', linestyle='dashed')

# prints a list of edges.
    def print_visibility_edges(self, visibility_edges, vertexes):

        # edge = initial vertex and a final/second vertex.
        a = None  # First vertex
        b = None  # Second vertex
        v_list = []
        for edge in visibility_edges:

          # for which vertex?
            counter = 0
            for i in range(0, len(vertexes)):

              # for ->  a=b=?

              # vertex(x,y)= edge's points (x,y)

                for j in vertexes[i]:
                    if j.x == edge.p1.x and j.y == edge.p1.y:
                        a = counter
                    counter += 1

            counter = 0

            for i in range(0, len(vertexes)):
                for j in vertexes[i]:
                    if j.x == edge.p2.x and j.y == edge.p2.y:
                        b = counter
                    counter += 1

            v_list.append((a, b))
        print( len(v_list), "visibility edges are below:")
        print(v_list)
        return v_list

# -------------------------------------------------------------------------------
# ------------------------------------- RPS -------------------------------------

class Implementation:
    def __init__(self, vertexes, edges):
        self.vertexes = vertexes
        self.obstacles_edges = edges

# separate vertexes
    def get_vertexes_from_dict(self, v_dict):
        vertexes = []

          # keys = v_dict
          # values =v_dict[i]

        for i in v_dict:
            for i in v_dict[i]:
                vertexes.append(i)
        return vertexes

#  For each vertex vi calculate αi (the angle from the horizontal axis to the line segment wi).
    def angle(self,y,x):
        angle = np.arctan2(y,x)

         # <0 = -ve angle -> + 2π = to make it +ve

        if angle < 0:
            angle = (angle + 2*np.pi)
        return angle

 # before sorting = save the prev list
    def copy_vertex_list(self,list):
      new_lst = []
      for vertex in list:
        new_lst.append(Point(vertex.x, vertex.y))  # Use the custom Point class
      return new_lst

# S edges list
    def S_initialization(self,half_line, current_vertex):

        S = []

        for edge in self.obstacles_edges:

          # intersecting pt
            is_intersect = half_line.intersect(edge)
            temp_point= half_line.intersection_point(edge)

             # is_intersect[0] = half line and edge intersection pt
             # and
             # dist.curr/start=dist.temp_point !=0
             # vertex intersection point is not the same as the current vertex.

            if (is_intersect[0] and round(current_vertex.dist(temp_point),0) != 0):
                edge.distance = current_vertex.dist(temp_point)
                S.append(edge)

        # line 14
        S = sorted(S, key=lambda x: x.distance)

        return S

# isVisible()

    def is_visible(self,v,vi,s, sweep_line):

        # line 8 -> 10
        if len(s) == 0:
            return True


        # If both v and vi lay on the same edge in S, vi is visible from v= line 1->4
        for i in s:

          #  v.dist=vi.dist == 0 (lying on the same INTERSECTING edge)
            if round(v.dist_segment(i),3) == 0. and round(vi.dist_segment(i),3) == 0.:
                return True


        # If vi and v are on the same obstacle and if the midpoint between them is inside the obstacle
        # vi is not visible from v = line 2 -> 4
        if self.inside_polygon(v,vi,s):
            return False


        # If the first edge in S intersect the sweepline going from v to vi, vi is not visible from v = line 11 -> 12
        for edge in s:
            is_interset = sweep_line.intersect(edge)


        # intersecting pt and v(curr/start) is NOT on the same INTERSECTING edge = OBSTACLE
            if is_interset[0] and not(round(v.dist_segment(edge),3) == 0.):
                return False
            else:
                return True

#  inside a polygon

    def inside_polygon(self, v, vi, s):

        #  both vertexes belong to same obstacle

        # id1/2 = obstacle in which v and vi belong = comes from .csv
        id1 = None   #first vertex
        id2 = None   # 2nd vertex

         # vertexes = # obstacles
         # vertexes[i] = vertexes in each obstacle

        for i in range(0,len(self.vertexes)):
            for j in self.vertexes[i]:

                # v, vi = belongs to same obstacle = store in id1, id2
                if (v.x,v.y) == (j.x,j.y):
                    id1 = i
                if (vi.x,vi.y) == (j.x,j.y):
                    id2 = i


         # if both vertexes belong to the same obstacle, the MP bw them is inside an obstacle, vi is not visible from v
        if id1 == id2:

          # create polygon
            poly_path = mplPath.Path(np.array([[vertex.x,vertex.y] for vertex in self.vertexes[id1]]))
            midpoint = ((v.x+vi.x)/2, (v.y+vi.y)/2)
            return poly_path.contains_point(midpoint)
        else:
            return False


    # rmv repeated edges = from final visibility graph

    def remove_repeated(self, visible):
        i = 0
        j = 1
        while i<len(visible) - 1:
            while j<len(visible):
                if (visible[i].p1.x == visible[j].p2.x and visible[i].p1.y == visible[j].p2.y and visible[i].p2.x == visible[j].p1.x and visible[i].p2.y == visible[j].p1.y) :
                    visible.remove(visible[j])
                    break
                j+=1
            i+=1
            j = i+1

        return [ x for x in visible if not(x.p1.x == x.p2.x and x.p1.y == x.p2.y)]

    # RPS
    def rotational_sweep(self):

        vertexes = self.get_vertexes_from_dict(self.vertexes)
        sorted_vertexes = self.copy_vertex_list(vertexes)
        visibility = []

        for k in range(0,len(vertexes)):
            v = vertexes[k] # Vertex = reference = start/curr

           # ε = sort vertex acc to angle
            for point in sorted_vertexes:
                point.alpha(self.angle(point.y-v.y,point.x-v.x))

            sorted_vertexes = sorted(sorted_vertexes, key=lambda x: x.alph)

            # create half line = 100 units
            half_line = Segment(v,Point(v.x+100,v.y))

            # begin S initialization
            S = self.S_initialization(half_line, vertexes[k])

   # start visibility checking of vi wrt v (start/curr)

            for vi in sorted_vertexes:

              # obstacle edges
                for edge in self.obstacles_edges:

                    # dist(edge)= dist(vi) =0
                    if round(vi.dist_segment(edge),2) == 0. and edge not in S:
                        S.append(edge)

                      # dist(edge)= dist(vi)= dist(v) =0
                    elif (round(vi.dist_segment(edge),2) == 0.  and edge in S) or (round(v.dist_segment(edge),2) == 0. and edge in S):
                        S.remove(edge)


                # MOVE LINE in anticlockwise direction =
                # sweep line from vertex v to vi with an angle offset of 0.001 and a magnitude of 100

                vi_SL = Point(v.x+(100)*np.cos(vi.alph + 0.001),v.y+(100)*np.sin(vi.alph + 0.001))  # Point (x,y)
                sweep_line = Segment(v,vi_SL) # point -> segment

                # Calculate the distance of the sweepline to every edge in S (obstacle edges)
                for s_edge in S:
                    temp_point= sweep_line.intersection_point(s_edge)
                    s_edge.distance = v.dist(temp_point)

                # Sort the S list with respect which obstacle edge is closer to v
                S = sorted(S, key=lambda x: x.distance)

                # potential edge
                sweep_line1 = Segment(v,vi)

                # Check for visibility
                if self.is_visible(v,vi,S, sweep_line1):
                    visibility.append(Segment(v,vi))


        return self.remove_repeated(visibility) # Return the visibility edges excluding repeated ones

# ----------------------------------------------------------------------------------
# ------------------------ MAIN ----------------------------------------------------

# visibility_graph.py
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import matplotlib.path as mplPath

# ... (The rest of your script)

if __name__ == "__main__":
    # Create an argument parser to accept the CSV file path
    parser = argparse.ArgumentParser(description="Visibility Graph Generator")
    parser.add_argument("csv_file", help="Path to the CSV environment file")
    args = parser.parse_args()

    # CSV path from the command-line argument
    csv_path = args.csv_file

    # Define all needed variables for building the visibility graph
    graph = Visibility(csv_path)

    # Get edges and vertexes from the CSV file
    E, vertexes = graph.get_edges_vertexes()

    # Step 1: Plot the environment's edges
    graph.plot_edges(E)

    # Perform Rotational Plane Sweep Algorithm (RPS)
    rps_algorithm = Implementation(vertexes, E)
    visibility_edges = rps_algorithm.rotational_sweep()

    # Step 2: Plot the visibility graph edges (potential edges)
    graph.plot_visibility(visibility_edges)

    # Step 3: Plot vertexes (in green)
    graph.plot_vertexes(vertexes)

    # Print the visibility edges as required
    visibility_edges_list = graph.print_visibility_edges(visibility_edges, vertexes)

    plt.savefig('visibility_graph.png')

    plt.show()
