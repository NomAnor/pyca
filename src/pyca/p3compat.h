#ifndef PYCA_P3COMPAT
#define PYCA_P3COMPAT
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#define DECLARE_INIT(name)  PyMODINIT_FUNC PyInit_##name(void)
#define INITERROR           return NULL
#define PyInt_Check         PyLong_Check
#define PyInt_CheckExact    PyLong_CheckExact
#define PyInt_AsLong        PyLong_AsLong
#define PyInt_FromLong      PyLong_FromLong
#define PyString_FromString PyUnicode_FromString
#define PyString_Check      PyUnicode_Check
#define PyString_FromFormat PyUnicode_FromFormat
// This should suffice, as long as we don't call this twice and try to hold onto both!
static char *PyString_AsString(PyObject *o)
{
    static char *result = NULL;
    if (result) {
        free(result);
        result = NULL;
    }
    if (PyUnicode_Check(o)) {
        PyObject *temp_bytes = PyUnicode_AsEncodedString(o, "ASCII", "strict");
        if (temp_bytes != NULL) {
            result = PyBytes_AS_STRING(temp_bytes);
            result = strdup(result);
            Py_DECREF(temp_bytes);
            return result;
        }
        return NULL;
    }
    return NULL;
}

static int PyString_Size(PyObject *o)
{
    if (PyUnicode_Check(o)) {
        PyObject *temp_bytes = PyUnicode_AsEncodedString(o, "ASCII", "strict");
        if (temp_bytes != NULL) {
            int result = PyBytes_GET_SIZE(temp_bytes);
            Py_DECREF(temp_bytes);
            return result;
        }
        return 0;
    }
    return 0;
}
#else
#define DECLARE_INIT(name)  PyMODINIT_FUNC init##name(void)
#define INITERROR           return
#define Py_TYPE(o)          ((o)->ob_type)
#define PyVarObject_HEAD_INIT(type, size)   PyObject_HEAD_INIT(type) size,
#endif
#endif