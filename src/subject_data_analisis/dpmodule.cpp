#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "DecisionPolicy.hpp"

/* method fpt(decisionPolicy, tolerance=1e-12) */
static PyObject* dpmod_xbounds(PyObject* self, PyObject* args, PyObject* keywds){
	double tolerance = 1e-12;
	PyObject* py_dp;
	PyObject* py_set_rho_in_py_dp = NULL;
	int set_rho_in_py_dp = 0;
	PyObject* py_out = NULL;
	PyObject* py_xub = NULL;
	PyObject* py_xlb = NULL;
	
	
	static char* kwlist[] = {"decPol", "tolerance","set_rho", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|dO", kwlist, &py_dp, &tolerance, &py_set_rho_in_py_dp))
		return NULL;
	
	if (tolerance <= 0.0) {
		PyErr_SetString(PyExc_ValueError, "tolerance needs to be larger than 0");
		return NULL;
	}
	set_rho_in_py_dp = PyObject_IsTrue(py_set_rho_in_py_dp);
	if (set_rho_in_py_dp==-1) { // Failed to evaluate truth statement
		PyErr_SetString(PyExc_ValueError, "set_rho needs to evaluate to a valid truth statemente");
		return NULL;
	}
	
	PyObject* py_model_var = PyObject_GetAttrString(py_dp,"model_var");
	PyObject* py_prior_mu_mean = PyObject_GetAttrString(py_dp,"prior_mu_mean");
	PyObject* py_prior_mu_var = PyObject_GetAttrString(py_dp,"prior_mu_var");
	PyObject* py_n = PyObject_GetAttrString(py_dp,"n");
	PyObject* py_dt = PyObject_GetAttrString(py_dp,"dt");
	PyObject* py_T = PyObject_GetAttrString(py_dp,"T");
	PyObject* py_reward = PyObject_GetAttrString(py_dp,"reward");
	PyObject* py_penalty = PyObject_GetAttrString(py_dp,"penalty");
	PyObject* py_iti = PyObject_GetAttrString(py_dp,"iti");
	PyObject* py_tp = PyObject_GetAttrString(py_dp,"tp");
	PyObject* py_cost = PyObject_GetAttrString(py_dp,"cost");

	if (py_model_var==NULL || py_prior_mu_mean==NULL || py_prior_mu_var==NULL ||
		py_n==NULL || py_dt==NULL || py_T==NULL || py_reward==NULL ||
		py_penalty==NULL || py_iti==NULL || py_tp==NULL || py_cost==NULL){
		PyErr_SetString(PyExc_ValueError, "Could not parse all decisionPolicy property values");
		Py_XDECREF(py_model_var);
		Py_XDECREF(py_prior_mu_mean);
		Py_XDECREF(py_prior_mu_var);
		Py_XDECREF(py_n);
		Py_XDECREF(py_dt);
		Py_XDECREF(py_T);
		Py_XDECREF(py_reward);
		Py_XDECREF(py_penalty);
		Py_XDECREF(py_iti);
		Py_XDECREF(py_tp);
		Py_XDECREF(py_cost);
		return NULL;
	}
	double model_var = PyFloat_AsDouble(py_model_var);
	double prior_mu_mean = PyFloat_AsDouble(py_prior_mu_mean);
	double prior_mu_var = PyFloat_AsDouble(py_prior_mu_var);
	int n = int(PyInt_AS_LONG(py_n));
	double dt = PyFloat_AsDouble(py_dt);
	double T = PyFloat_AsDouble(py_T);
	double reward = PyFloat_AsDouble(py_reward);
	double penalty = PyFloat_AsDouble(py_penalty);
	double iti = PyFloat_AsDouble(py_iti);
	double tp = PyFloat_AsDouble(py_tp);
	double cost = PyFloat_AsDouble(py_cost);
	// Check if an error occured while getting the c typed values from the python objects
	if (PyErr_Occurred()!=NULL){
		Py_XDECREF(py_model_var);
		Py_XDECREF(py_prior_mu_mean);
		Py_XDECREF(py_prior_mu_var);
		Py_XDECREF(py_n);
		Py_XDECREF(py_dt);
		Py_XDECREF(py_T);
		Py_XDECREF(py_reward);
		Py_XDECREF(py_penalty);
		Py_XDECREF(py_iti);
		Py_XDECREF(py_tp);
		Py_XDECREF(py_cost);
		return NULL;
	}

	DecisionPolicy dp = DecisionPolicy(model_var, prior_mu_mean, prior_mu_var, n, dt, T,
							reward, penalty, iti, tp, cost);

	py_out = PyTuple_New(2);
	npy_intp py_nT[1] = { dp.nT };
	py_xub = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
	py_xlb = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);

	if (py_out==NULL || py_xub==NULL || py_xlb==NULL){
		PyErr_SetString(PyExc_MemoryError, "Out of memory");
		goto error_cleanup;
	}

	dp.iterate_rho_value(tolerance);
	if (set_rho_in_py_dp){
		if (PyObject_SetAttrString(py_dp,"rho",Py_BuildValue("d",dp.rho))==-1){
			PyErr_SetString(PyExc_ValueError, "Could not set decisionPolicy property rho");
			goto error_cleanup;
		}
	}
	dp.x_ubound((double*) PyArray_DATA((PyArrayObject*) py_xub));
	dp.x_lbound((double*) PyArray_DATA((PyArrayObject*) py_xlb));
	PyTuple_SET_ITEM(py_out, 0, py_xub); // Steals a reference to py_xub so no dec_ref must be called on py_xub on cleanup
	PyTuple_SET_ITEM(py_out, 1, py_xlb); // Steals a reference to py_xlb so no dec_ref must be called on py_xlb on cleanup
	
	// normal_cleanup
	Py_XDECREF(py_model_var);
	Py_XDECREF(py_prior_mu_mean);
	Py_XDECREF(py_prior_mu_var);
	Py_XDECREF(py_n);
	Py_XDECREF(py_dt);
	Py_XDECREF(py_T);
	Py_XDECREF(py_reward);
	Py_XDECREF(py_penalty);
	Py_XDECREF(py_iti);
	Py_XDECREF(py_tp);
	Py_XDECREF(py_cost);
	return py_out;

error_cleanup:
	Py_XDECREF(py_model_var);
	Py_XDECREF(py_prior_mu_mean);
	Py_XDECREF(py_prior_mu_var);
	Py_XDECREF(py_n);
	Py_XDECREF(py_dt);
	Py_XDECREF(py_T);
	Py_XDECREF(py_reward);
	Py_XDECREF(py_penalty);
	Py_XDECREF(py_iti);
	Py_XDECREF(py_tp);
	Py_XDECREF(py_cost);
	Py_XDECREF(py_xub);
	Py_XDECREF(py_xlb);
	Py_XDECREF(py_out);
	return NULL;
}

static PyMethodDef DPMethods[] = {
    {"xbounds", (PyCFunction) dpmod_xbounds, METH_VARARGS | METH_KEYWORDS,
     "Computes the decision bounds in x(t) space (i.e. the accumulated sensory input space)\n\n  (xub, xlb) = xbounds(dp, tolerance=1e-12)\n\nComputes the decision bounds for a decisionPolicy instance specified in 'dp'.\nThis function is more memory and computationally efficient than calling dp.invert_belief();dp.value_dp(); xb = dp.belief_bound_to_x_bound(b); from python. Another difference is that this function returns a tuple of (upper_bound, lower_bound) instead of a numpy array whose first element is upper_bound and second element is lowe_bound.\n'tolerance' is a float that indicates the tolerance when searching for the rho value that yields value[int(n/2)]=0."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
    /* module initialisation for Python 3 */
    static struct PyModuleDef dpmodule = {
       PyModuleDef_HEAD_INIT,
       "dp",   /* name of module */
       "Module to compute the decision bounds for bayesian inference",
       -1,
       DPMethods
    };

    PyMODINIT_FUNC PyInit_dp(void)
    {
        PyObject *m = PyModule_Create(&dpmodule);
        import_array();
        return m;
    }
#else
    /* module initialisation for Python 2 */
    PyMODINIT_FUNC initdp(void)
    {
        Py_InitModule("dp", DPMethods);
        import_array();
    }
#endif
