#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <structmember.h>
#include <map>

#include <cadef.h>
#include <alarm.h>

#include <pthread.h>
#include <assert.h>
#include <stdlib.h>

#include "p3compat.h"
#include "pyca.hh"
#include "getfunctions.hh"
#include "putfunctions.hh"
#include "handlers.hh"

extern "C" {
    //
    // Python methods for channel access PV types
    //
    static PyObject* create_channel(PyObject* self, PyObject*)
    {
        capv* pv = reinterpret_cast<capv*>(self);
        if (pv->cid) {
            pyca_raise_pyexc_pv("create_channel", "channel already created", pv);
        }
        const int capriority = 10;
        int result = ca_create_channel(pv->name,
                                       pyca_connection_handler,
                                       self,
                                       capriority,
                                       &pv->cid);
        if (result != ECA_NORMAL) {
            pyca_raise_caexc_pv("ca_create_channel", result, pv);
        }
        Py_RETURN_NONE;
    }

    static PyObject* clear_channel(PyObject* self, PyObject*)
    {
        capv* pv = reinterpret_cast<capv*>(self);

        chid cid = pv->cid;
        if (!cid) {
            pyca_raise_pyexc_pv("clear_channel", "channel is null", pv);
        }
        int result;
        Py_BEGIN_ALLOW_THREADS
            result = ca_clear_channel(cid);
        Py_END_ALLOW_THREADS
        if (result != ECA_NORMAL) {
            pyca_raise_caexc_pv("ca_clear_channel", result, pv);
        }
        pv->cid = 0;
        Py_RETURN_NONE;
    }

    static PyObject* subscribe_channel(PyObject* self, PyObject* args)
    {
        capv* pv = reinterpret_cast<capv*>(self);
        unsigned long event_mask;
        int control;
        int count = -1;

        if (!PyArg_ParseTuple(args, "ki|i:subscribe_channel", &event_mask, &control, &count)) return NULL;

        if (pv->simulated) {
            if (control) {
                pyca_raise_pyexc_pv("subscribe_channel", "Can't get control info on simulated PV", pv);
            }
            pv->count = 0;
            if (count >= 0) {
                pv->count = count;
            }
            pv->didmon = 1;
            Py_RETURN_NONE;
        }

        chid cid = pv->cid;
        if (!cid) {
            pyca_raise_pyexc_pv("subscribe_channel", "channel is null", pv);
        }
        pv->count = ca_element_count(cid);
        if (count >= 0 && count < pv->count) {
            pv->count = count;
        }
        short type = ca_field_type(cid);
        if (pv->count == 0 || type == TYPENOTCONN) {
            pyca_raise_caexc_pv("ca_field_type", ECA_DISCONNCHID, pv);
        }
        short dbr_type = control ?
            dbf_type_to_DBR_CTRL(type) : // Asks IOC to send status+time+limits+value
            dbf_type_to_DBR_TIME(type);  // Asks IOC to send status+time+value
        if (dbr_type_is_ENUM(dbr_type) && pv->string_enum) {
            dbr_type = control ? DBR_CTRL_STRING : DBR_TIME_STRING;
        }

        int result = ca_create_subscription(dbr_type,
                                            pv->count,
                                            cid,
                                            event_mask,
                                            pyca_monitor_handler,
                                            pv,
                                            &pv->eid);
        if (result != ECA_NORMAL) {
            pyca_raise_caexc_pv("ca_create_subscription", result, pv);
        }
        Py_RETURN_NONE;
    }

    static PyObject* unsubscribe_channel(PyObject* self, PyObject*)
    {
        capv* pv = reinterpret_cast<capv*>(self);

        if (pv->simulated) {
            pv->didmon = 0;
            Py_RETURN_NONE;
        }

        evid eid = pv->eid;
        if (eid) {
            int result = ca_clear_subscription(eid);
            if (result != ECA_NORMAL) {
                pyca_raise_caexc_pv("ca_clear_subscription", result, pv);
            }
            pv->eid = 0;
        }
        Py_RETURN_NONE;
    }

    static PyObject* replace_access_rights_event(PyObject* self, PyObject*)
    {
        capv* pv = reinterpret_cast<capv*>(self);
        chid cid = pv->cid;
        if (!cid) {
            pyca_raise_pyexc_pv("replace_access_rights_event", "channel is null", pv);
        }
        int result = ca_replace_access_rights_event(cid, pyca_access_rights_handler);
        if (result != ECA_NORMAL) {
            pyca_raise_caexc_pv("replace_access_rights_event", result, pv);
        }
        Py_RETURN_NONE;
    }

    static PyObject* get_enum_strings(PyObject* self, PyObject* pytmo)
    {
        capv* pv = reinterpret_cast<capv*>(self);

        chid cid = pv->cid;
        if (!cid) {
            pyca_raise_pyexc_pv("get_enum_strings", "channel is null", pv);
        }

        short type = ca_field_type(cid);
        if (type == TYPENOTCONN) {
            pyca_raise_caexc_pv("ca_field_type", ECA_DISCONNCHID, pv);
        }

        if (!dbr_type_is_ENUM(dbf_type_to_DBR(type))) {
            pyca_raise_pyexc_pv("get_enum_strings", "channel is not ENUM type", pv);
        }
        int result;
        double timeout = PyFloat_AsDouble(pytmo);
        if (PyErr_Occurred()) return NULL;
        if (timeout < 0) {
            result = ca_array_get_callback(DBR_GR_ENUM,
                                           1,
                                           cid,
                                           pyca_getevent_handler,
                                           pv);
            if (result != ECA_NORMAL) {
                pyca_raise_caexc_pv("ca_array_get_callback", result, pv);
            }
        } else {
            struct dbr_gr_enum buffer;
            result = ca_array_get (DBR_GR_ENUM, 1, cid, &buffer);
            if (result != ECA_NORMAL) {
                pyca_raise_caexc_pv("ca_array_get", result, pv);
            }
            Py_BEGIN_ALLOW_THREADS
                result = ca_pend_io(timeout);
            Py_END_ALLOW_THREADS
            if (result != ECA_NORMAL) {
                pyca_raise_caexc_pv("ca_pend_io", result, pv);
            }
            if (_pyca_event_process(pv, &buffer, DBR_GR_ENUM, 1) != 0) return NULL;
        }
        Py_RETURN_NONE;
    }

    static PyObject* get_data(PyObject* self, PyObject* args)
    {
        capv* pv = reinterpret_cast<capv*>(self);
        int control;
        double timeout;
        int count = -1;

        if (!PyArg_ParseTuple(args, "id|i:get_data", &control, &timeout, &count)) return NULL;

        if (pv->simulated) {
            if (control) {
                pyca_raise_pyexc_pv("get_data", "Can't get control info on simulated PV", pv);
            }
            pv->count = 0;
            if (count >= 0) {
                pv->count = count;
            }
            if (timeout > 0) {
                pyca_raise_pyexc_pv("get_data", "Can't specify a  get timeout on simulated PV", pv);
            }
            pv->didget = 1;
            Py_RETURN_NONE;
        }

        chid cid = pv->cid;
        if (!cid) {
            pyca_raise_pyexc_pv("get_data", "channel is null", pv);
        }
        pv->count = ca_element_count(cid);
        if (count >= 0 && count < pv->count) {
            pv->count = count;
        }
        short type = ca_field_type(cid);
        if (pv->count == 0 || type == TYPENOTCONN) {
            pyca_raise_caexc_pv("ca_field_type", ECA_DISCONNCHID, pv);
        }
        short dbr_type = control ?
            dbf_type_to_DBR_CTRL(type) : // Asks IOC to send status+time+limits+value
            dbf_type_to_DBR_TIME(type);  // Asks IOC to send status+time+value
        if (dbr_type_is_ENUM(dbr_type) && pv->string_enum) {
            dbr_type = control ? DBR_CTRL_STRING : DBR_TIME_STRING;
        }
        if (timeout < 0) {
            int result = ca_array_get_callback(dbr_type,
                                               pv->count,
                                               cid,
                                               pyca_getevent_handler,
                                               pv);
            if (result != ECA_NORMAL) {
                pyca_raise_caexc_pv("ca_array_get_callback", result, pv);
            }
        } else {
            void* buffer = _pyca_adjust_buffer_size(pv, dbr_type, pv->count, 0);
            if (!buffer) return NULL;
            int result = ca_array_get(dbr_type,
                                      pv->count,
                                      cid,
                                      buffer);
            if (result != ECA_NORMAL) {
                pyca_raise_caexc_pv("ca_array_get", result, pv);
            }
            Py_BEGIN_ALLOW_THREADS
                result = ca_pend_io(timeout);
            Py_END_ALLOW_THREADS
            if (result != ECA_NORMAL) {
                pyca_raise_caexc_pv("ca_pend_io", result, pv);
            }
            if (_pyca_event_process(pv, buffer, dbr_type, pv->count) != 0) return NULL;
        }
        Py_RETURN_NONE;
    }

    static PyObject* put_data(PyObject* self, PyObject* args)
    {
        capv* pv = reinterpret_cast<capv*>(self);
        PyObject* pyval;
        double timeout;

        if (!PyArg_ParseTuple(args, "Od:put_data", &pyval, &timeout)) return NULL;

        chid cid = pv->cid;
        if (!cid) {
            pyca_raise_pyexc_pv("put_data", "channel is null", pv);
        }
        int count = ca_element_count(cid);
        short type = ca_field_type(cid);
        if (count == 0 || type == TYPENOTCONN) {
            pyca_raise_caexc_pv("ca_field_type", ECA_DISCONNCHID, pv);
        }
        if (count > 1) {
            if (PyList_Check(pyval)) {
                int tcnt = PyList_GET_SIZE(pyval);
                if (tcnt < count) {
                    count = tcnt;
                }
            }
        }
        short dbr_type = dbf_type_to_DBR(type);
        const void* buffer = _pyca_put_buffer(pv, pyval, dbr_type, count);
        if (!buffer) return NULL;
        if (timeout < 0) {
            int result = ca_array_put_callback(dbr_type,
                                               count,
                                               cid,
                                               buffer,
                                               pyca_putevent_handler,
                                               pv);
            if (result != ECA_NORMAL) {
                pyca_raise_caexc_pv("ca_array_put_callback", result, pv);
            }
        } else {
            int result = ca_array_put(dbr_type,
                                      count,
                                      cid,
                                      buffer);
            if (result != ECA_NORMAL) {
                pyca_raise_caexc_pv("ca_array_put", result, pv);
            }
            Py_BEGIN_ALLOW_THREADS
                result = ca_pend_io(timeout);
            Py_END_ALLOW_THREADS
            if (result != ECA_NORMAL) {
                pyca_raise_caexc_pv("ca_pend_io", result, pv);
            }
        }
        Py_RETURN_NONE;
    }

    static PyObject* host(PyObject* self, PyObject*)
    {
        capv* pv = reinterpret_cast<capv*>(self);

        if (!pv->cid) pyca_raise_pyexc_pv("host", "channel is null", pv);
        return PyString_FromString(ca_host_name(pv->cid));
    }

    static PyObject* state(PyObject* self, PyObject*)
    {
        capv* pv = reinterpret_cast<capv*>(self);

        if (!pv->cid) pyca_raise_pyexc_pv("state", "channel is null", pv);
        return PyInt_FromLong(ca_state(pv->cid));
    }

    static PyObject* count(PyObject* self, PyObject*)
    {
        capv* pv = reinterpret_cast<capv*>(self);

        if (!pv->cid) pyca_raise_pyexc_pv("count", "channel is null", pv);
        return PyInt_FromLong(ca_element_count(pv->cid));
    }

    static PyObject* type(PyObject* self, PyObject*)
    {
        capv* pv = reinterpret_cast<capv*>(self);

        if (!pv->cid) pyca_raise_pyexc_pv("type", "channel is null", pv);
        return PyString_FromString(dbf_type_to_text(ca_field_type(pv->cid)));
    }

    static PyObject* rwaccess(PyObject* self, PyObject*)
    {
        capv* pv = reinterpret_cast<capv*>(self);

        if (!pv->cid) pyca_raise_pyexc_pv("rwaccess", "channel is null", pv);
        int rw = ca_read_access(pv->cid) ? 1 : 0;
        rw |= ca_write_access(pv->cid) ? 2 : 0;
        return PyInt_FromLong(rw);
    }

    static PyObject* is_string_enum(PyObject* self, PyObject*)
    {
        capv* pv = reinterpret_cast<capv*>(self);
        return PyBool_FromLong(pv->string_enum);
    }

    static PyObject* set_string_enum(PyObject* self, PyObject* pyval)
    {
        capv* pv = reinterpret_cast<capv*>(self);

        pv->string_enum = PyObject_IsTrue(pyval);
        Py_RETURN_NONE;
    }

    static bool numpy_arrays = false;

    // Built-in methods for the capv type
    static int capv_init(PyObject* self, PyObject* args, PyObject* kwds)
    {
        capv* pv = reinterpret_cast<capv*>(self);
        char const *c_name;
        if (!PyArg_ParseTuple(args, "s:capv_init", &c_name)) return -1;

        pv->name = strdup(c_name);
        pv->processor = NULL;
        pv->connect_cb = NULL;
        pv->monitor_cb = NULL;
        pv->rwaccess_cb = NULL;
        pv->getevt_cb = NULL;
        pv->putevt_cb = NULL;
        pv->simulated = 0;
        pv->use_numpy = numpy_arrays;
        pv->cid = 0;
        pv->getbuffer = 0;
        pv->getbufsiz = 0;
        pv->putbuffer = 0;
        pv->putbufsiz = 0;
        pv->eid = 0;
        return 0;
    }

    static void capv_dealloc(PyObject* self)
    {
        capv* pv = reinterpret_cast<capv*>(self);
        free(pv->name);
        Py_XDECREF(pv->data);
        Py_XDECREF(pv->processor);
        Py_XDECREF(pv->connect_cb);
        Py_XDECREF(pv->monitor_cb);
        Py_XDECREF(pv->rwaccess_cb);
        Py_XDECREF(pv->getevt_cb);
        Py_XDECREF(pv->putevt_cb);
        if (pv->cid) {
            ca_clear_channel(pv->cid);
            pv->cid = 0;
        }
        if (pv->getbuffer) {
            delete [] pv->getbuffer;
            pv->getbuffer = 0;
            pv->getbufsiz = 0;
        }
        if (pv->putbuffer) {
            delete [] pv->putbuffer;
            pv->putbuffer = 0;
            pv->putbufsiz = 0;
        }
        Py_TYPE(self)->tp_free(self);
    }

    static PyObject* capv_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
    {
        PyObject* self = type->tp_alloc(type, 0);
        capv* pv = reinterpret_cast<capv*>(self);
        if (!pv) {
            pyca_raise_pyexc("capv_new", "cannot allocate new PV");
        }
        pv->data = PyDict_New();
        if (!pv->data) {
            pyca_raise_pyexc("capv_new", "cannot allocate dictionary for new PV");
        }
        Py_INCREF(pv);
        return self;
    }

    // Register capv methods
    static PyMethodDef capv_methods[] = {
        {"create_channel",              create_channel,              METH_NOARGS},
        {"clear_channel",               clear_channel,               METH_NOARGS},
        {"subscribe_channel",           subscribe_channel,           METH_VARARGS},
        {"unsubscribe_channel",         unsubscribe_channel,         METH_NOARGS},
        {"get_data",                    get_data,                    METH_VARARGS},
        {"put_data",                    put_data,                    METH_VARARGS},
        {"host",                        host,                        METH_NOARGS},
        {"state",                       state,                       METH_NOARGS},
        {"count",                       count,                       METH_NOARGS},
        {"type",                        type,                        METH_NOARGS},
        {"rwaccess",                    rwaccess,                    METH_NOARGS},
        {"replace_access_rights_event", replace_access_rights_event, METH_NOARGS},
        {"set_string_enum",             set_string_enum,             METH_O},
        {"is_string_enum",              is_string_enum,              METH_NOARGS},
        {"get_enum_strings",            get_enum_strings,            METH_O},
        {NULL},
    };

    // Register capv members
    static PyMemberDef capv_members[] = {
        {"name",        T_STRING,    offsetof(capv, name),        1, "name"}, // READONLY
        {"data",        T_OBJECT_EX, offsetof(capv, data),        1, "data"}, // READONLY
        {"processor",   T_OBJECT_EX, offsetof(capv, processor),   0, "processor"},
        {"connect_cb",  T_OBJECT_EX, offsetof(capv, connect_cb),  0, "connect_cb"},
        {"monitor_cb",  T_OBJECT_EX, offsetof(capv, monitor_cb),  0, "monitor_cb"},
        {"rwaccess_cb", T_OBJECT_EX, offsetof(capv, rwaccess_cb), 0, "rwaccess_cb"},
        {"getevt_cb",   T_OBJECT_EX, offsetof(capv, getevt_cb),   0, "getevt_cb"},
        {"putevt_cb",   T_OBJECT_EX, offsetof(capv, putevt_cb),   0, "putevt_cb"},
        {"simulated",   T_BOOL,      offsetof(capv, simulated),   0, "simulated"},
        {"use_numpy",   T_BOOL,      offsetof(capv, use_numpy),   0, "use_numpy"},
        {NULL}
    };

    static PyTypeObject capv_type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "pyca.capv",                               /* tp_name */
        sizeof(capv),                              /* tp_basicsize */
        0,                                         /* tp_itemsize */
        capv_dealloc,                              /* tp_dealloc */
        NULL,                                      /* tp_print */
        NULL,                                      /* tp_getattr */
        NULL,                                      /* tp_setattr */
        NULL,                                      /* tp_as_async */
        NULL,                                      /* tp_repr */
        NULL,                                      /* tp_as_number */
        NULL,                                      /* tp_as_sequence */
        NULL,                                      /* tp_as_mapping */
        NULL,                                      /* tp_hash */
        NULL,                                      /* tp_call */
        NULL,                                      /* tp_str */
        NULL,                                      /* tp_getattro */
        NULL,                                      /* tp_setattro */
        NULL,                                      /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */
        NULL,                                      /* tp_doc */
        NULL,                                      /* tp_traverse */
        NULL,                                      /* tp_clear */
        NULL,                                      /* tp_richcompare */
        0,                                         /* tp_weaklistoffset */
        NULL,                                      /* tp_iter */
        NULL,                                      /* tp_iternext */
        capv_methods,                              /* tp_methods */
        capv_members,                              /* tp_members */
        NULL,                                      /* tp_getset */
        NULL,                                      /* tp_base */
        NULL,                                      /* tp_dict */
        NULL,                                      /* tp_descr_get */
        NULL,                                      /* tp_descr_set */
        0,                                         /* tp_dictoffset */
        capv_init,                                 /* tp_init */
        NULL,                                      /* tp_alloc */
        capv_new,                                  /* tp_new */
    };

    // Module functions
    static PyObject* initialize(PyObject*, PyObject*)
    {
        //     PyEval_InitThreads();
        //     int result = ca_context_create(ca_enable_preemptive_callback);
        //     if (result != ECA_NORMAL) {
        //       pyca_raise_caexc("ca_context_create", result);
        //     }
        printf("warning: no need to invoke initialize with this version of pyca\n");
        Py_RETURN_NONE;
    }

    static PyObject* finalize(PyObject*, PyObject*)
    {
        //     ca_context_destroy();
        printf("warning: no need to invoke finalize with this version of pyca\n");
        Py_RETURN_NONE;
    }

    // Each process needs a unique context.
    static std::map<pid_t, ca_client_context*> ca_context_map;

    static bool has_proc_context()
    {
        return ca_context_map.count(::getpid()) == 1;
    }

    static void save_proc_context()
    {
        ca_context_map[::getpid()] = ca_current_context();
    }

    static ca_client_context* get_proc_context()
    {
        return ca_context_map[::getpid()];
    }

    // Each thread needs the same context as the process that spawned it
    static PyObject* attach_context(PyObject* self, PyObject* args)
    {
        // only failure modes are if it's already attached or single threaded,
        // so no need to raise an exception
        if (ca_current_context() == NULL) {
            if (!has_proc_context()) {
                pyca_raise_pyexc("attach_context", "no context to attach");
            }
            int result = ca_attach_context(get_proc_context());
            if (result != ECA_NORMAL) {
                pyca_raise_pyexc("attach_context", "attach error");
            }
        }
        Py_RETURN_NONE;
    }

    static PyObject* new_context(PyObject*, PyObject*)
    {
        // use to create context for multiprocessing module
        // if this process already has a context, skip
        if (!has_proc_context()) {
            ca_detach_context();
            int result = ca_context_create(ca_enable_preemptive_callback);
            if (result != ECA_NORMAL) {
                pyca_raise_caexc("ca_context_create", result);
            }
            save_proc_context();
        }
        Py_RETURN_NONE;
    }

    static PyObject* pend_io(PyObject*, PyObject* pytmo)
    {
        int result;
        double timeout = PyFloat_AsDouble(pytmo);
        if (PyErr_Occurred()) return NULL;
        Py_BEGIN_ALLOW_THREADS
            result = ca_pend_io(timeout);
        Py_END_ALLOW_THREADS
        if (result != ECA_NORMAL) {
            pyca_raise_caexc("ca_pend_io", result);
        }
        Py_RETURN_NONE;
    }

    static PyObject* flush_io(PyObject*, PyObject*)
    {
        int result = ca_flush_io();
        if (result != ECA_NORMAL) {
            pyca_raise_caexc("ca_flush_io", result);
        }
        Py_RETURN_NONE;
    }

    static PyObject* pend_event(PyObject*, PyObject* pytmo)
    {
        int result;
        double timeout = PyFloat_AsDouble(pytmo);
        if (PyErr_Occurred()) return NULL;
        Py_BEGIN_ALLOW_THREADS
            result = ca_pend_event(timeout);
        Py_END_ALLOW_THREADS
        if (result != ECA_TIMEOUT) {
            pyca_raise_caexc("ca_pend_event", result);
        }
        Py_RETURN_NONE;
    }

    static PyObject* set_numpy(PyObject*, PyObject* np)
    {
        numpy_arrays = PyObject_IsTrue(np);
        Py_RETURN_NONE;
    }

    // Register module methods
    static PyMethodDef pyca_methods[] = {
        {"attach_context", attach_context, METH_NOARGS},
        {"new_context",    new_context,    METH_NOARGS},
        {"initialize",     initialize,     METH_NOARGS},
        {"finalize",       finalize,       METH_NOARGS},
        {"pend_io",        pend_io,        METH_O},
        {"flush_io",       flush_io,       METH_NOARGS},
        {"pend_event",     pend_event,     METH_O},
        {"set_numpy",      set_numpy,      METH_O},
        {NULL}
    };

    static const char* AlarmSeverityStrings[ALARM_NSEV] = {
        "NO_ALARM", "MINOR", "MAJOR", "INVALID"
    };

    static const char* AlarmConditionStrings[ALARM_NSTATUS] = {
        "NO_ALARM",
        "READ_ALARM",
        "WRITE_ALARM",
        "HIHI_ALARM",
        "HIGH_ALARM",
        "LOLO_ALARM",
        "LOW_ALARM",
        "STATE_ALARM",
        "COS_ALARM",
        "COMM_ALARM",
        "TIMEOUT_ALARM",
        "HWLIMIT_ALARM",
        "CALC_ALARM",
        "SCAN_ALARM",
        "LINK_ALARM",
        "SOFT_ALARM",
        "BAD_SUB_ALARM",
        "UDF_ALARM",
        "DISABLE_ALARM",
        "SIMM_ALARM",
        "READ_ACCESS_ALARM",
        "WRITE_ACCESS_ALARM",
    };

#ifdef IS_PY3K
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "pyca",
        NULL,
        -1,
        pyca_methods
    };
#endif

    // Initialize python module
#ifdef IS_PY3K
    PyMODINIT_FUNC PyInit_pyca(void)
#else
    PyMODINIT_FUNC initpyca(void)
#endif
    {
        // initialize numpy C API
        import_array();

        // initialize tp_base and ob_type fields
        if (PyType_Ready(&capv_type) < 0) {
#ifdef IS_PY3K
            return NULL;
#else
            return;
#endif
        }

#ifdef IS_PY3K
        PyObject* module = PyModule_Create(&moduledef);
#else
        PyObject* module = Py_InitModule("pyca", pyca_methods);
#endif

        if (module == NULL) {
#ifdef IS_PY3K
            return NULL;
#else
            return;
#endif
        }

        PyObject* v = NULL;

        // Export selected channel access constants
        if (PyModule_AddIntConstant(module, "DBE_VALUE", DBE_VALUE) != 0) goto error;
        if (PyModule_AddIntConstant(module, "DBE_LOG", DBE_LOG) != 0) goto error;
        if (PyModule_AddIntConstant(module, "DBE_ALARM", DBE_ALARM) != 0) goto error;
        v = PyTuple_New(ALARM_NSEV);
        if (!v) goto error;
        for (unsigned i = 0; i < ALARM_NSEV; i++) {
            if (PyModule_AddIntConstant(module, AlarmSeverityStrings[i], i) != 0) {
                Py_DECREF(v);
                goto error;
            }
            PyTuple_SET_ITEM(v, i, PyString_FromString(AlarmSeverityStrings[i]));
        }
        if (PyModule_AddObject(module, "severity", v) != 0) goto error;
        v = PyTuple_New(ALARM_NSTATUS);
        if (!v) goto error;
        for (unsigned i = 0; i < ALARM_NSTATUS; i++) {
            if (PyModule_AddIntConstant(module, AlarmConditionStrings[i], i) != 0) {
                Py_DECREF(v);
                goto error;
            }
            PyTuple_SET_ITEM(v, i, PyString_FromString(AlarmConditionStrings[i]));
        }
        if (PyModule_AddObject(module, "alarm", v) != 0) goto error;
        // secs between Jan 1st 1970 and Jan 1st 1990
        if (PyModule_AddIntConstant(module, "epoch", 7305 * 86400) != 0) goto error;

        v = PyCapsule_New((void *)pyca_getevent_handler, "pyca.get_handler", NULL);
        if (!v) goto error;
        if (PyModule_AddObject(module, "get_handler", v) != 0) goto error;
        v = PyCapsule_New((void *)pyca_monitor_handler, "pyca.monitor_handler", NULL);
        if (!v) goto error;
        if (PyModule_AddObject(module, "monitor_handler", v) != 0) goto error;

        // Add capv type to this module
        Py_INCREF(&capv_type);
        if (PyModule_AddObject(module, "capv", (PyObject*)&capv_type) != 0) goto error;

        // Add custom exceptions to this module
        pyca_pyexc = PyErr_NewException("pyca.pyexc", NULL, NULL);
        if (!pyca_pyexc) goto error;
        if (PyModule_AddObject(module, "pyexc", pyca_pyexc) != 0) goto error;
        pyca_caexc = PyErr_NewException("pyca.caexc", NULL, NULL);
        if (!pyca_caexc) goto error;
        if (PyModule_AddObject(module, "caexc", pyca_caexc) != 0) goto error;
        // incref for the references held by our c code
        Py_INCREF(pyca_pyexc);
        Py_INCREF(pyca_caexc);

        // libca creates non-python threads so we must ensure that python threading is initialized
        PyEval_InitThreads();
        if (!has_proc_context()) {
            int result = ca_context_create(ca_enable_preemptive_callback);
            if (result != ECA_NORMAL) {
                fprintf(stderr,
                        "*** initpyca: ca_context_create failed with status %d\n",
                        result);
                goto error;
            }
            save_proc_context();
            // The following seems to cause a segfault at exit
            //Py_AtExit(ca_context_destroy);
        }
#ifdef IS_PY3K
        return module;
#endif

error:
#ifdef IS_PY3K
        Py_DECREF(module);
        return NULL;
#else
        return;
#endif
    }
}
