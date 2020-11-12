# Travelling salesman problem  https://developers.google.com/optimization/routing/tsp
import numpy as np
from scipy.spatial import distance_matrix
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def get_route(manager, routing, assignment):
    index = routing.Start(0)
    route = []

    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = assignment.Value(routing.NextVar(index))

    return np.array(route)


def solve_tsp(points, dist_mat=None, time_limit=10,
              verbose=1):
    """
    Get the index list for the optimal route for all points, starting at the first
    :param points:
    :param dist_mat: optional
    :param time_limit: seconds
    :param verbose:
    :return:
    """

    n = len(points)
    if dist_mat is None:
        dist_mat = distance_matrix(points, points)

    if dist_mat.dtype != np.integer:
        min_dist = dist_mat[dist_mat != 0].min()
        if min_dist < 1:
            dist_mat /= min_dist

    # Create the routing index manager. num_cities, num_vehicles, depot
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_mat[from_node, to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # Additional search parameters
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = time_limit

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    route = get_route(manager=manager, routing=routing, assignment=assignment)

    if verbose:
        cost = dist_mat[route, np.roll(route, - 1)].sum()
        print(f"TSP Cost for {points.shape} points after {time_limit}s: {cost}")

    return route
