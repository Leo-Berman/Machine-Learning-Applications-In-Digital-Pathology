import shapely

def generate_polygon(coords):
    shape = shapely.Polygon(coords)
    return shape

def get_border(shape):
    return (shape.exterior.xy)

