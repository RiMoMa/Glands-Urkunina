from shapely.geometry import Polygon, MultiPolygon
def select_random_points_in_polygon(polygon, num_points=3, left=0, top=0):
    """
    Selecciona puntos aleatorios dentro del polígono ajustado al sistema de referencia del parche.

    Parameters:
        polygon (shapely.geometry.Polygon): Polígono de la anomalía.
        num_points (int): Número de puntos aleatorios a seleccionar.
        left (int): Coordenada X de la esquina superior izquierda del parche en el WSI.
        top (int): Coordenada Y de la esquina superior izquierda del parche en el WSI.

    Returns:
        list: Lista de puntos aleatorios en el espacio del parche.
    """
    from shapely.geometry import Point
    import random

    # Ajustar el polígono al sistema de referencia del parche
    adjusted_polygon = Polygon([(x - left, y - top) for x, y in polygon.exterior.coords])

    # Seleccionar puntos aleatorios dentro del polígono ajustado
    points = []
    min_x, min_y, max_x, max_y = adjusted_polygon.bounds

    while len(points) < num_points:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if adjusted_polygon.contains(random_point):
            points.append((random_point.x, random_point.y))

    return points