#ifndef PYCA_P3COMPAT
#define PYCA_P3COMPAT

#if PY_MAJOR_VERSION >= 3

#define IS_PY3K
#define PyInt_Check         PyLong_Check
#define PyInt_CheckExact    PyLong_CheckExact
#define PyInt_AsLong        PyLong_AsLong
#define PyInt_FromLong      PyLong_FromLong
#define PyString_FromString PyUnicode_FromString
#define PyString_AsString   PyUnicode_AsUTF8
#define PyString_Check      PyUnicode_Check
#define PyString_FromFormat PyUnicode_FromFormat

#endif

#endif
