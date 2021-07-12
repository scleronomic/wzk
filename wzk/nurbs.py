# TODO implement NURBS with numpy, focus on quick evaluation + derivative
#   https://public.vrac.iastate.edu/~oliver/courses/me625/week8.pdf

# from geomdl import NURBS
# from geomdl import knotvector
# from geomdl.visualization import VisMPL
#
#
# # Create a 3-dimensional B-spline Curve
# curve = NURBS.Curve()
# NURBS.
# # Set degree
# curve.degree = 2
#
# # Set control points (weights vector will be 1 by default)
# # Use curve.ctrlptsw is if you are using homogeneous points as Pw
# curve.ctrlpts = [[10, 5, 10, 1], [10, 20, -30, 1], [40, 10, 25, 1], [-10, 5, 0, 1 ]]
# # curve.ctrlptsw = [[10, 5, 10], [10, 20, -30], [40, 10, 25], [-10, 5, 0]]
#
# # Set knot vector
# curve.knotvector = knotvector.generate(degree=curve.degree, num_ctrlpts=curve.ctrlpts_size)
#
# # Set evaluation delta (controls the number of curve points)
# curve.delta = 0.01
#
# # Get curve points (the curve will be automatically evaluated)
# curve_points = curve.evalpts
#
# curve.vis = VisMPL.VisCurve3D()
# curve.render()
