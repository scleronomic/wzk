#include <CGAL/Cartesian_d.h>
#include <CGAL/Min_sphere_of_spheres_d.h>
#include <vector>
#include <Python.h>

#define PY_SSIZE_T_CLEAN

typedef CGAL::Cartesian_d<float> K;
typedef K::Point_d Point;

const int D2 = 2;
const int D3 = 3;
const int D4 = 4;
typedef float coord2[D2];
typedef float coord3[D3];
typedef float coord4[D4];
typedef CGAL::Min_sphere_of_spheres_d_traits_d<K, float, D2> Traits2;
typedef CGAL::Min_sphere_of_spheres_d_traits_d<K, float, D3> Traits3;
typedef CGAL::Min_sphere_of_spheres_d_traits_d<K, float, D4> Traits4;
typedef CGAL::Min_sphere_of_spheres_d<Traits2> Min_sphere2;
typedef CGAL::Min_sphere_of_spheres_d<Traits3> Min_sphere3;
typedef CGAL::Min_sphere_of_spheres_d<Traits4> Min_sphere4;
typedef Traits2::Sphere Sphere2;
typedef Traits3::Sphere Sphere3;
typedef Traits4::Sphere Sphere4;


PyObject * min_sphere2_py(PyObject * self, PyObject * args, PyObject * kwargs) {

    // initialize buffer for arguments
    PyObject * x;
    PyObject * r;
    PyObject * res;
    int n;
    int d=2;

    Py_buffer x_v;
    Py_buffer r_v;
    Py_buffer res_v;
    static char * keywords[] = {"x", "r", "n", "res", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "OOiO", keywords, &x, &r, &n, &res);

    PyObject_GetBuffer(x, &x_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(r, &r_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(res, &res_v, PyBUF_SIMPLE);

    std::vector<Sphere2> S;
    coord2 * x_p = (coord2 *)x_v.buf;

    float * r_p = (float *)r_v.buf;
    float * res_p = (float *)res_v.buf;

    for (int i=0; i<n; ++i) {
        Point p(d, x_p[i], x_p[i]+d);
        S.push_back(Sphere2(p, r_p[i]));
    }

    Min_sphere2 ms(S.begin(), S.end());

    for (int i=0; i<d; ++i)
        res_p[i] = (float)ms.center_cartesian_begin()[i];
    res_p[d] = (float)ms.radius();

    // release buffer
    PyBuffer_Release(&x_v);
    PyBuffer_Release(&r_v);
    PyBuffer_Release(&res_v);
    Py_RETURN_NONE;
}


PyObject * min_sphere3_py(PyObject * self, PyObject * args, PyObject * kwargs) {

    // initialize buffer for arguments
    PyObject * x;
    PyObject * r;
    PyObject * res;
    int n;
    int d = 3;

    Py_buffer x_v;
    Py_buffer r_v;
    Py_buffer res_v;
    static char * keywords[] = {"x", "r", "n", "res", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "OOiO", keywords, &x, &r, &n, &res);

    PyObject_GetBuffer(x, &x_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(r, &r_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(res, &res_v, PyBUF_SIMPLE);

    std::vector<Sphere3> S;
    coord3 * x_p = (coord3 *)x_v.buf;
    float * r_p = (float *)r_v.buf;
    float * res_p = (float *)res_v.buf;

    for (int i=0; i<n; ++i) {
        Point p(d, x_p[i], x_p[i]+d);
        S.push_back(Sphere3(p, r_p[i]));
    }

    Min_sphere3 ms(S.begin(), S.end());

    for (int i=0; i<d; ++i)
        res_p[i] = (float)ms.center_cartesian_begin()[i];
    res_p[d] = (float)ms.radius();

    // release buffer
    PyBuffer_Release(&x_v);
    PyBuffer_Release(&r_v);
    PyBuffer_Release(&res_v);
    Py_RETURN_NONE;
}


PyObject * min_sphere4_py(PyObject * self, PyObject * args, PyObject * kwargs) {

    // initialize buffer for arguments
    PyObject * x;
    PyObject * r;
    PyObject * res;
    int n;
    int d = 4;

    Py_buffer x_v;
    Py_buffer r_v;
    Py_buffer res_v;
    static char * keywords[] = {"x", "r", "n", "res", NULL};
    PyArg_ParseTupleAndKeywords(args, kwargs, "OOiO", keywords, &x, &r, &n, &res);

    PyObject_GetBuffer(x, &x_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(r, &r_v, PyBUF_SIMPLE);
    PyObject_GetBuffer(res, &res_v, PyBUF_SIMPLE);

    std::vector<Sphere4> S;
    coord4 * x_p = (coord4 *)x_v.buf;
    float * r_p = (float *)r_v.buf;
    float * res_p = (float *)res_v.buf;

    for (int i=0; i<n; ++i) {
        Point p(d, x_p[i], x_p[i]+d);
        S.push_back(Sphere4(p, r_p[i]));
    }

    Min_sphere4 ms(S.begin(), S.end());

    for (int i=0; i<d; ++i)
        res_p[i] = (float)ms.center_cartesian_begin()[i];
    res_p[d] = (float)ms.radius();

    // release buffer
    PyBuffer_Release(&x_v);
    PyBuffer_Release(&r_v);
    PyBuffer_Release(&res_v);
    Py_RETURN_NONE;
}


PyMethodDef module_methods[] = {
    {"min_sphere2", (PyCFunction)min_sphere2_py, METH_VARARGS | METH_KEYWORDS, NULL},
    {"min_sphere3", (PyCFunction)min_sphere3_py, METH_VARARGS | METH_KEYWORDS, NULL},
    {"min_sphere4", (PyCFunction)min_sphere4_py, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL},
};

PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "MinSphere", NULL, -1, module_methods};

extern "C" PyObject * PyInit_MinSphere() {
    PyObject * module = PyModule_Create(&module_def);
    return module;
}
