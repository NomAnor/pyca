#include "p3compat.h"

// Channel access PUT template functions
// These return 0 on success or -1 on failure
static inline int _pyca_put(PyObject* pyvalue, dbr_string_t* buf)
{
    char *result = PyString_AsString(pyvalue);
    if (!result) {
        (*buf)[0] = 0;
        return -1;
    }

    memcpy(buf, result, sizeof(dbr_string_t));
    return 0;
}

static inline int _pyca_put(PyObject* pyvalue, dbr_enum_t* buf)
{
    long result = PyInt_AsLong(pyvalue);
    if (PyErr_Occurred()) return -1;

    *buf = result;
    return 0;
}

static inline int _pyca_put(PyObject* pyvalue, dbr_char_t* buf)
{
    long result = PyInt_AsLong(pyvalue);
    if (PyErr_Occurred()) return -1;

    *buf = result;
    return 0;
}

static inline int _pyca_put(PyObject* pyvalue, dbr_short_t* buf)
{
    long result = PyInt_AsLong(pyvalue);
    if (PyErr_Occurred()) return -1;

    *buf = result;
    return 0;
}

static inline int _pyca_put(PyObject* pyvalue, dbr_long_t* buf)
{
    long result = PyInt_AsLong(pyvalue);
    if (PyErr_Occurred()) return -1;

    *buf = result;
    return 0;
}

static inline int _pyca_put(PyObject* pyvalue, dbr_float_t* buf)
{
    double result = PyFloat_AsDouble(pyvalue);
    if (PyErr_Occurred()) return -1;

    *buf = result;
    return 0;
}

static inline int _pyca_put(PyObject* pyvalue, dbr_double_t* buf)
{
    double result = PyFloat_AsDouble(pyvalue);
    if (PyErr_Occurred()) return -1;

    *buf = result;
    return 0;
}

// If using correct numpy type, there is nothing special to do.
// Note we still risk corrupt data if we mismatch numpy data types, but this
// should never happen unless the python user is malicious.
template<class T> static inline
int _pyca_put_np(void* npdata, T* buf)
{
    memcpy(buf, npdata, sizeof(T));
    return 0;
}

// Copy python objects into channel access void* buffer
// Return the buffer on success or NULL on failure
template<class T> static inline
void* _pyca_put_value(capv* pv, PyObject* pyvalue, long count)
{
    unsigned size = count*sizeof(T);
    if (size != pv->putbufsiz) {
        delete [] pv->putbuffer;
        pv->putbuffer = new char[size];
        pv->putbufsiz = size;
    }
    T* buffer = reinterpret_cast<T*>(pv->putbuffer);
    if (count == 1) {
        if (PyList_Check(pyvalue)) {
            PyObject* pyval = PyList_GetItem(pyvalue, 0);
            if (!pyval) return NULL;

            if (_pyca_put(pyval, buffer) != 0) return NULL;
        } else if (PyArray_Check(pyvalue)) {
            void* npdata = PyArray_GETPTR1(pyvalue, 0);

            if (PyArray_IsPythonScalar(pyvalue)) {
                PyObject* pyval = PyArray_GETITEM(pyvalue, npdata);

                if (_pyca_put(pyval, buffer) != 0) return NULL;
            } else {
                if (_pyca_put_np(npdata, buffer) != 0) return NULL;
            }
        } else {
            if (_pyca_put(pyvalue, buffer) != 0) return NULL;
        }
    } else {
        if (PyList_Check(pyvalue)) {
            for (long i = 0; i < count; i++) {
                PyObject* pyval = PyList_GetItem(pyvalue, i);
                if (!pyval) return NULL;

                if (_pyca_put(pyval, buffer+i) != 0) return NULL;
            }
        } else if (PyArray_Check(pyvalue)) {
            bool py_type = PyArray_IsPythonScalar(pyvalue);
            for (long i = 0; i < count; i++) {
                void* npdata = PyArray_GETPTR1(pyvalue, i);

                if (py_type) {
                    PyObject* pyval = PyArray_GETITEM(pyvalue, npdata);

                    if (_pyca_put(pyval, buffer+i) != 0) return NULL;
                } else {
                    if (_pyca_put_np(npdata, buffer+i) != 0) return NULL;
                }
            }
        }
    }
    return buffer;
}

// Return the buffer on success or NULL on failure
static const void* _pyca_put_buffer(capv* pv,
                                    PyObject* pyvalue,
                                    short &dbr_type, // We may change DBF_ENUM to DBF_STRING
                                    long count)
{
    switch (dbr_type) {
    case DBR_ENUM:
        {
            if (PyString_Check(pyvalue)) {
                dbr_type = DBR_STRING;
                // no break: pass into string block, below
                // Note: We don't check if the caller passed
                //       an integer cast as a string... Although
                //       this seems to be handled correctly.
            } else {
                return _pyca_put_value<dbr_enum_t>(pv, pyvalue, count);
            }
        }
    case DBR_STRING:
        return _pyca_put_value<dbr_string_t>(pv, pyvalue, count);
    case DBR_CHAR:
        return _pyca_put_value<dbr_char_t>(pv, pyvalue, count);
    case DBR_SHORT:
        return _pyca_put_value<dbr_short_t>(pv, pyvalue, count);
    case DBR_LONG:
        return _pyca_put_value<dbr_long_t>(pv, pyvalue, count);
    case DBR_FLOAT:
        return _pyca_put_value<dbr_float_t>(pv, pyvalue, count);
    case DBR_DOUBLE:
        return _pyca_put_value<dbr_double_t>(pv, pyvalue, count);
    }
    pyca_raise_pyexc_pv("_pyca_put_buffer", "un-handled type", pv);
}
