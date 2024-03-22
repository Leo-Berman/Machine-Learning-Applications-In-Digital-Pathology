import shapely

# given a set of coordinates this will return a shapley shape data type of
# those coordinates
#
def generate_polygon(coords):
    shape = shapely.Polygon(coords)
    return shape

# given a shape it will return a list of lists
# that can be used to plot in matplotlib
#
def get_border(shape):
    return (shape.exterior.xy)

'''
ex. that creates a 10x10 square and plots a rnadom squiggle within
the square and save it as an imgage
def create_grid():
    square_coords = [(0,0),(0,10),(10,10),(10,0)]
    square = generate_polygon(square_coords)
    square_x,square_y = get_border(square)
    
    amorphous_coords=[(2,2),(3,4),(4,4),(5,5),(2,1)]
    amorphous = generate_polygon(amorphous_coords)
    amorphous_x,amorphous_y = get_border(amorphous)
    plt.plot(square_x,square_y)
    plt.plot(amorphous_x,amorphous_y)
    plt.savefig("DATA/demo.png")
'''

'''
# ex. that will plot a 1x1 square with the bottom right corner at the origin
# shape = generate_polygone([[0,0,],[0,1],[1,1],[1,0]])
# points = get_border(shape)
# matplotlib.pyplot.plot(points[0],points[1])
'''
