#include "p3compat.h"

// Utility function which steals the reference to val and adds it to dict
// Returns 0 on success or -1 on failure.
static inline int _pyca_setitem(PyObject* dict, const char* key, PyObject* val)
{
    if (!val) {
        PyErr_SetString(PyExc_ValueError, "No value object");
        return -1;
    }
    if (!dict || !key) {
        Py_DECREF(val);
        PyErr_SetString(PyExc_ValueError, "No dict object or key object");
        return -1;
    }

    int result = PyDict_SetItemString(dict, key, val);
    Py_DECREF(val);
    return result;
}

// Channel access GET template functions
// These return new references or NULL on failure.
static inline PyObject* _pyca_get(const dbr_string_t value)
{
    return PyString_FromString(value);
}

static inline PyObject* _pyca_get(const dbr_enum_t value)
{
    return PyInt_FromLong(value);
}

static inline PyObject* _pyca_get(const dbr_char_t value)
{
    return PyInt_FromLong(value);
}

static inline PyObject* _pyca_get(const dbr_short_t value)
{
    return PyInt_FromLong(value);
}

static inline PyObject* _pyca_get(const dbr_ulong_t value)
{
    return PyInt_FromLong(value);
}

static inline PyObject* _pyca_get(const dbr_long_t value)
{
    return PyInt_FromLong(value);
}

static inline PyObject* _pyca_get(const dbr_float_t value)
{
    return PyFloat_FromDouble(value);
}

static inline PyObject* _pyca_get(const dbr_double_t value)
{
    return PyFloat_FromDouble(value);
}

// Copy the value of a channel access object into a python
// object. Note that for array objects (count > 1), this function will
// check for the presence of a user process method ('processor') and,
// if present, that method is invoked. This mechanism may avoid
// unncessary data copies and it may be useful for large arrays.
// Returns 0 on success or -1 on failure.
typedef int (*processptr)(const void* cadata, long count, size_t size, void* descr);

// EPICS       Description                 Numpy
// DBR_STRING  40 character string`        NPY_STRING
// DBR_ENUM    16-bit unsigned integer     NPY_UINT16
// DBR_CHAR    8-bit character             NPY_UINT8
// DBR_SHORT   16-bit integer              NPY_INT16
// DBR_LONG    32-bit signed integer       NPY_INT32
// DBR_FLOAT   32-bit IEEE floating point  NPY_FLOAT32
// DBR_DOUBLE  64-bit IEEE floating pint   NPY_FLOAT64
int _numpy_array_type(const dbr_string_t*)
{
    return NPY_STRING;
}

int _numpy_array_type(const dbr_enum_t*)
{
    return NPY_UINT16;
}

int _numpy_array_type(const dbr_char_t*)
{
    return NPY_UINT8;
}

int _numpy_array_type(const dbr_short_t*)
{
    return NPY_INT16;
}

int _numpy_array_type(const dbr_ulong_t*)
{
    return NPY_UINT32;
}

int _numpy_array_type(const dbr_long_t*)
{
    return NPY_INT32;
}

int _numpy_array_type(const dbr_float_t*)
{
    return NPY_FLOAT32;
}

int _numpy_array_type(const dbr_double_t*)
{
    return NPY_FLOAT64;
}

// Return a new reference or NULL on failure
template<class T> static inline
PyObject* _pyca_get_value(capv* pv, const T* dbrv, long count)
{
    if (count == 1) {
        return _pyca_get(dbrv->value);
    } else {
        if (!pv->processor) {
            if (pv->use_numpy) {
                npy_intp dims[1] = {count};
                int typenum = _numpy_array_type(&(dbrv->value));

                PyObject* nparray = PyArray_EMPTY(1, dims, typenum, 0);
                if (!nparray) return NULL;

                memcpy(PyArray_DATA(nparray), &(dbrv->value), count*sizeof(dbrv->value));
                return nparray;
            } else {
                PyObject* pylist = PyList_New(count);
                if (!pylist) return NULL;

                for (long i = 0; i < count; i++) {
                    PyObject* item = _pyca_get(*(&(dbrv->value)+i));
                    if (!item) {
                        Py_DECREF(pylist);
                        return NULL;
                    }
                    PyList_SET_ITEM(pylist, i, item);
                }
                return pylist;
            }
        } else {
            const char* name = PyCapsule_GetName(pv->processor);
            if (PyErr_Occurred()) return NULL;

            processptr process = (processptr)PyCapsule_GetPointer(pv->processor, name);
            if (!process) return NULL;

            void* descr = PyCapsule_GetContext(pv->processor);
            if (PyErr_Occurred()) return NULL;

            if (process(&(dbrv->value), count, sizeof(dbrv->value), descr) != 0) {
                PyErr_SetString(PyExc_RuntimeError, "Processor function failed");
                return NULL;
            }
        }
    }
    Py_RETURN_NONE;
}

// Copy channel access status objects into python
// Return 0 on success or -1 on failure
template<class T> static inline
int _pyca_get_sts(capv* pv, const T* dbrv, long count)
{
    PyObject* pydata = pv->data;
    if (_pyca_setitem(pydata, "status",   _pyca_get(dbrv->status))          != 0) return -1;
    if (_pyca_setitem(pydata, "severity", _pyca_get(dbrv->severity))        != 0) return -1;
    if (_pyca_setitem(pydata, "value",    _pyca_get_value(pv, dbrv, count)) != 0) return -1;

    return 0;
}

// Copy channel access time objects into python
// Return 0 on success or -1 on failure
template<class T> static inline
int _pyca_get_time(capv* pv, const T* dbrv, long count)
{
    PyObject* pydata = pv->data;
    if (_pyca_setitem(pydata, "status",   _pyca_get(dbrv->status))             != 0) return -1;
    if (_pyca_setitem(pydata, "severity", _pyca_get(dbrv->severity))           != 0) return -1;
    if (_pyca_setitem(pydata, "secs",     _pyca_get(dbrv->stamp.secPastEpoch)) != 0) return -1;
    if (_pyca_setitem(pydata, "nsec",     _pyca_get(dbrv->stamp.nsec))         != 0) return -1;
    if (_pyca_setitem(pydata, "value",    _pyca_get_value(pv, dbrv, count))    != 0) return -1;

    return 0;
}

// Copy channel access control objects into python
// Return 0 on success or -1 on failure
template<class T> static inline
int _pyca_get_ctrl_long(capv* pv, const T* dbrv, long count)
{
    PyObject* pydata = pv->data;
    if (_pyca_setitem(pydata, "status",       _pyca_get(dbrv->status))              != 0) return -1;
    if (_pyca_setitem(pydata, "severity",     _pyca_get(dbrv->severity))            != 0) return -1;
    if (_pyca_setitem(pydata, "units",        _pyca_get(dbrv->units))               != 0) return -1;
    if (_pyca_setitem(pydata, "display_llim", _pyca_get(dbrv->lower_disp_limit))    != 0) return -1;
    if (_pyca_setitem(pydata, "display_hlim", _pyca_get(dbrv->upper_disp_limit))    != 0) return -1;
    if (_pyca_setitem(pydata, "warn_llim",    _pyca_get(dbrv->lower_warning_limit)) != 0) return -1;
    if (_pyca_setitem(pydata, "warn_hlim",    _pyca_get(dbrv->upper_warning_limit)) != 0) return -1;
    if (_pyca_setitem(pydata, "alarm_llim",   _pyca_get(dbrv->lower_alarm_limit))   != 0) return -1;
    if (_pyca_setitem(pydata, "alarm_hlim",   _pyca_get(dbrv->upper_alarm_limit))   != 0) return -1;
    if (_pyca_setitem(pydata, "ctrl_llim",    _pyca_get(dbrv->lower_ctrl_limit))    != 0) return -1;
    if (_pyca_setitem(pydata, "ctrl_hlim",    _pyca_get(dbrv->upper_ctrl_limit))    != 0) return -1;
    if (_pyca_setitem(pydata, "value",        _pyca_get_value(pv, dbrv, count))     != 0) return -1;

    return 0;
}

// Return 0 on success or -1 on failure
template<class T> static inline
int _pyca_get_ctrl_enum(capv* pv, const T* dbrv, long count)
{
    PyObject* pydata = pv->data;
    if (_pyca_setitem(pydata, "status",   _pyca_get(dbrv->status))          != 0) return -1;
    if (_pyca_setitem(pydata, "severity", _pyca_get(dbrv->severity))        != 0) return -1;
    if (_pyca_setitem(pydata, "no_str",   _pyca_get(dbrv->no_str))          != 0) return -1;
    if (_pyca_setitem(pydata, "value",    _pyca_get_value(pv, dbrv, count)) != 0) return -1;

    return 0;
}

// Return 0 on success or -1 on failure
template<class T> static inline
int _pyca_get_ctrl_double(capv* pv, const T* dbrv, long count)
{
    PyObject* pydata = pv->data;
    if (_pyca_setitem(pydata, "status",       _pyca_get(dbrv->status))              != 0) return -1;
    if (_pyca_setitem(pydata, "severity",     _pyca_get(dbrv->severity))            != 0) return -1;
    if (_pyca_setitem(pydata, "precision",    _pyca_get(dbrv->precision))           != 0) return -1;
    if (_pyca_setitem(pydata, "units",        _pyca_get(dbrv->units))               != 0) return -1;
    if (_pyca_setitem(pydata, "display_llim", _pyca_get(dbrv->lower_disp_limit))    != 0) return -1;
    if (_pyca_setitem(pydata, "display_hlim", _pyca_get(dbrv->upper_disp_limit))    != 0) return -1;
    if (_pyca_setitem(pydata, "warn_llim",    _pyca_get(dbrv->lower_warning_limit)) != 0) return -1;
    if (_pyca_setitem(pydata, "warn_hlim",    _pyca_get(dbrv->upper_warning_limit)) != 0) return -1;
    if (_pyca_setitem(pydata, "alarm_llim",   _pyca_get(dbrv->lower_alarm_limit))   != 0) return -1;
    if (_pyca_setitem(pydata, "alarm_hlim",   _pyca_get(dbrv->upper_alarm_limit))   != 0) return -1;
    if (_pyca_setitem(pydata, "ctrl_llim",    _pyca_get(dbrv->lower_ctrl_limit))    != 0) return -1;
    if (_pyca_setitem(pydata, "ctrl_hlim",    _pyca_get(dbrv->upper_ctrl_limit))    != 0) return -1;
    if (_pyca_setitem(pydata, "value",        _pyca_get_value(pv, dbrv, count))     != 0) return -1;

    return 0;
}

// Return 0 on success or -1 on failure
static inline
int _pyca_get_gr_enum(capv* pv, const struct dbr_gr_enum* dbrv, long count)
{
    PyObject* pydata = pv->data;

    PyObject* enstrs = PyTuple_New(dbrv->no_str);
    if (!enstrs) return -1;

    for (int i = 0; i < dbrv->no_str; i++) {
        // TODO:  This is no bueno. We need a new accessor above for
        //        char arrays...
        PyObject* item = _pyca_get(dbrv->strs[i]);
        if (!item) {
            Py_DECREF(enstrs);
            return -1;
        }
        PyTuple_SET_ITEM(enstrs, i, item);
    }

    if (_pyca_setitem(pydata, "enum_set", enstrs) != 0) return -1;

    return 0;
}

// Return 0 on success or -1 on failure
static const int _pyca_event_process(capv* pv,
                                       const void* buffer,
                                       short dbr_type,
                                       long count)
{
    const db_access_val* dbr = reinterpret_cast<const db_access_val*>(buffer);
    if (!dbr) {
        PyErr_SetString(PyExc_BufferError, "Invalid buffer");
        return -1;
    }

    switch (dbr_type) {
    case DBR_GR_ENUM:
        return _pyca_get_gr_enum(pv, &dbr->genmval, count);
    case DBR_TIME_STRING:
        return _pyca_get_time(pv, &dbr->tstrval, count);
    case DBR_TIME_ENUM:
        return _pyca_get_time(pv, &dbr->tenmval, count);
    case DBR_TIME_CHAR:
        return _pyca_get_time(pv, &dbr->tchrval, count);
    case DBR_TIME_SHORT:
        return _pyca_get_time(pv, &dbr->tshrtval, count);
    case DBR_TIME_LONG:
        return _pyca_get_time(pv, &dbr->tlngval, count);
    case DBR_TIME_FLOAT:
        return _pyca_get_time(pv, &dbr->tfltval, count);
    case DBR_TIME_DOUBLE:
        return _pyca_get_time(pv, &dbr->tdblval, count);
    case DBR_CTRL_STRING:
        return _pyca_get_sts(pv, &dbr->sstrval, count);
    case DBR_CTRL_ENUM:
        return _pyca_get_ctrl_enum(pv, &dbr->cenmval, count);
    case DBR_CTRL_CHAR:
        return _pyca_get_ctrl_long(pv, &dbr->cchrval, count);
    case DBR_CTRL_SHORT:
        return _pyca_get_ctrl_long(pv, &dbr->cshrtval, count);
    case DBR_CTRL_LONG:
        return _pyca_get_ctrl_long(pv, &dbr->clngval, count);
    case DBR_CTRL_FLOAT:
        return _pyca_get_ctrl_double(pv, &dbr->cfltval, count);
    case DBR_CTRL_DOUBLE:
        return _pyca_get_ctrl_double(pv, &dbr->cdblval, count);
    }
    pyca_raise_pyexc_int("_pyca_event_process", "un-handled type", pv);
}

// Return the buffer on success or NULL on failure
static void* _pyca_adjust_buffer_size(capv* pv,
                                      short dbr_type,
                                      long count,
                                      int nxtbuf)
{
    unsigned size = 0;
    switch (dbr_type) {
    case DBR_TIME_STRING:
        size = sizeof(dbr_time_string) + sizeof(dbr_string_t)*(count-1);
        break;
    case DBR_TIME_ENUM:
        size = sizeof(dbr_time_enum) + sizeof(dbr_enum_t)*(count-1);
        break;
    case DBR_TIME_CHAR:
        size = sizeof(dbr_time_char) + sizeof(dbr_char_t)*(count-1);
        break;
    case DBR_TIME_SHORT:
        size = sizeof(dbr_time_short) + sizeof(dbr_short_t)*(count-1);
        break;
    case DBR_TIME_LONG:
        size = sizeof(dbr_time_long) + sizeof(dbr_long_t)*(count-1);
        break;
    case DBR_TIME_FLOAT:
        size = sizeof(dbr_time_float) + sizeof(dbr_float_t)*(count-1);
        break;
    case DBR_TIME_DOUBLE:
        size = sizeof(dbr_time_double) + sizeof(dbr_double_t)*(count-1);
        break;
    case DBR_CTRL_STRING:
        size = sizeof(dbr_sts_string) + sizeof(dbr_string_t)*(count-1);
        break;
    case DBR_CTRL_ENUM:
        size = sizeof(dbr_ctrl_enum) + sizeof(dbr_enum_t)*(count-1);
        break;
    case DBR_CTRL_CHAR:
        size = sizeof(dbr_ctrl_char) + sizeof(dbr_char_t)*(count-1);
        break;
    case DBR_CTRL_SHORT:
        size = sizeof(dbr_ctrl_short) + sizeof(dbr_short_t)*(count-1);
        break;
    case DBR_CTRL_LONG:
        size = sizeof(dbr_ctrl_long) + sizeof(dbr_long_t)*(count-1);
        break;
    case DBR_CTRL_FLOAT:
        size = sizeof(dbr_ctrl_float) + sizeof(dbr_float_t)*(count-1);
        break;
    case DBR_CTRL_DOUBLE:
        size = sizeof(dbr_ctrl_double) + sizeof(dbr_double_t)*(count-1);
        break;
    default:
        pyca_raise_pyexc_pv("_pyca_adjust_buffer_size", "un-handled type", pv);
    }
    if (size != pv->getbufsiz) {
        delete [] pv->getbuffer;
        pv->getbuffer = new char[size];
        pv->getbufsiz = size;
    }
    return pv->getbuffer;
}
