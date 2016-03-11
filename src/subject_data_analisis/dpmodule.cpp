#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "DecisionPolicy.hpp"

#include <cstdio>

/* method fpt(decisionPolicy, tolerance=1e-12) */
static PyObject* dpmod_xbounds(PyObject* self, PyObject* args, PyObject* keywds){
	double tolerance = 1e-12;
	PyObject* py_dp;
	PyObject* py_set_rho_in_py_dp = Py_None;
	PyObject* py_touch_py_bounds = Py_None;
	int set_rho_in_py_dp = 0;
	int touch_py_bounds = 0;
	int must_dec_ref_py_bounds = 1;
	bool must_create_py_bounds = false;
	PyObject* py_out = NULL;
	PyObject* py_xub = NULL;
	PyObject* py_xlb = NULL;
	DecisionPolicy* dp;
	
	static char* kwlist[] = {"decPol", "tolerance","set_rho","set_bounds", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|dOO", kwlist, &py_dp, &tolerance, &py_set_rho_in_py_dp, &py_touch_py_bounds))
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
	touch_py_bounds = PyObject_IsTrue(py_touch_py_bounds);
	if (touch_py_bounds==-1) { // Failed to evaluate truth statement
		PyErr_SetString(PyExc_ValueError, "set_bounds needs to evaluate to a valid truth statemente");
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
	PyObject* py_bounds = NULL;

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
	int nT = (int)(T/dt)+1;
	npy_intp py_nT[1] = {nT};
	double reward = PyFloat_AsDouble(py_reward);
	double penalty = PyFloat_AsDouble(py_penalty);
	double iti = PyFloat_AsDouble(py_iti);
	double tp = PyFloat_AsDouble(py_tp);
	double cost = 0.;
	if (PyObject_IsInstance(py_cost,(PyObject*)(&PyArray_Type))){
		if (!PyArray_IsAnyScalar((PyArrayObject*)py_cost)){
			PyErr_WarnEx(PyExc_RuntimeWarning,"dp module's function xbounds is only capable of handling integer cost values. It will assume that the cost is constant and equal to the first element of the supplied cost array.",1);
		}
		if (!PyArray_ISFLOAT((PyArrayObject*)py_cost)){
			PyErr_SetString(PyExc_ValueError,"Supplied cost must be a floating point number that can be casted to double.");
		} else {
			cost = ((double*)PyArray_DATA((PyArrayObject*)py_cost))[0];
		}
	} else {
		cost = PyFloat_AsDouble(py_cost);
	}
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
	
	if (!touch_py_bounds){
		dp = new DecisionPolicy(model_var, prior_mu_mean, prior_mu_var, n, dt, T,
								reward, penalty, iti, tp, cost);
	} else {
		npy_intp shape[2] = {2,nT};
		py_bounds = PyObject_GetAttrString(py_dp,"bounds");
		if (py_bounds==NULL){ // If the attribute bounds does not exist, create it
			PyErr_Clear(); // As we are handling the exception that py_dp has no attribute "bounds", we clear the exception state.
			must_create_py_bounds = true;
		} else if (!PyArray_Check((PyArrayObject*)py_bounds)){
			// Attribute 'bounds' in DecisionPolicy instance is not a numpy array. We must re create py_bounds
			Py_DECREF(py_bounds);
			must_create_py_bounds = true;
		} else if (PyArray_NDIM((PyArrayObject*)py_bounds)!=2){
			// Attribute 'bounds' in DecisionPolicy instance does not have the correct shape. We must re create py_bounds
			Py_DECREF(py_bounds);
			must_create_py_bounds = true;
		} else {
			for (int i=0;i<2;i++){
				if (shape[i]!=PyArray_SHAPE((PyArrayObject*)py_bounds)[i]){
					// Attribute 'bounds' in DecisionPolicy instance does not have the correct shape. We must re create py_bounds
					Py_DECREF(py_bounds);
					must_create_py_bounds = true;
					break;
				}
			}
		}
		if (must_create_py_bounds){
			py_bounds = PyArray_SimpleNew(2,shape,NPY_DOUBLE);
			if (py_bounds==NULL){
				PyErr_SetString(PyExc_MemoryError,"An error occured attempting to create the numpy array that would stores the DecisionPolicy instance's bounds attribute. Out of memory.");
				goto error_cleanup;
			}
			if (PyObject_SetAttrString(py_dp,"bounds", py_bounds)==-1){
				PyErr_SetString(PyExc_AttributeError,"Could not create and assign attribute bounds for the Decision policy instance");
				goto error_cleanup;
			}
			must_dec_ref_py_bounds = 0;
		}
		dp = new DecisionPolicy(model_var, prior_mu_mean, prior_mu_var, n, dt, T,
								reward, penalty, iti, tp, cost,
								(double*)PyArray_GETPTR2((PyArrayObject*)py_bounds,(npy_intp)0,(npy_intp)0),
								(double*)PyArray_GETPTR2((PyArrayObject*)py_bounds,(npy_intp)1,(npy_intp)0),
								((int) PyArray_STRIDE((PyArrayObject*)py_bounds,1))/sizeof(double)); // We divide by sizeof(double) because strides determines the number of bytes to stride in each dimension. As we cast the supplied void pointer to double*, each element has sizeof(double) bytes instead of 1 byte.
	}
	
	py_out = PyTuple_New(2);
	py_xub = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
	py_xlb = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);

	if (py_out==NULL || py_xub==NULL || py_xlb==NULL){
		PyErr_SetString(PyExc_MemoryError, "Out of memory");
		delete(dp);
		goto error_cleanup;
	}
	
	dp->iterate_rho_value(tolerance);
	if (set_rho_in_py_dp){
		if (PyObject_SetAttrString(py_dp,"rho",Py_BuildValue("d",dp->rho))==-1){
			PyErr_SetString(PyExc_ValueError, "Could not set decisionPolicy property rho");
			delete(dp);
			goto error_cleanup;
		}
	}
	// Backpropagate and compute bounds in the diffusion space
	dp->backpropagate_value();
	dp->x_ubound((double*) PyArray_DATA((PyArrayObject*) py_xub));
	dp->x_lbound((double*) PyArray_DATA((PyArrayObject*) py_xlb));
	PyTuple_SET_ITEM(py_out, 0, py_xub); // Steals a reference to py_xub so no dec_ref must be called on py_xub on cleanup
	PyTuple_SET_ITEM(py_out, 1, py_xlb); // Steals a reference to py_xlb so no dec_ref must be called on py_xlb on cleanup
	
	// normal_cleanup
	delete(dp);
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
	if (must_dec_ref_py_bounds) Py_XDECREF(py_bounds);
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
	if (must_dec_ref_py_bounds) Py_XDECREF(py_bounds);
	Py_XDECREF(py_xub);
	Py_XDECREF(py_xlb);
	Py_XDECREF(py_out);
	return NULL;
}

static PyMethodDef DPMethods[] = {
    {"xbounds", (PyCFunction) dpmod_xbounds, METH_VARARGS | METH_KEYWORDS,
     "Computes the decision bounds in x(t) space (i.e. the accumulated sensory input space)\n\n  (xub, xlb) = xbounds(dp, tolerance=1e-12, set_rho=None, set_bounds=None)\n\nComputes the decision bounds for a decisionPolicy instance specified in 'dp'.\nThis function is more memory and computationally efficient than calling dp.invert_belief();dp.value_dp(); xb = dp.belief_bound_to_x_bound(b); from python. Another difference is that this function returns a tuple of (upper_bound, lower_bound) instead of a numpy array whose first element is upper_bound and second element is lowe_bound.\n'tolerance' is a float that indicates the tolerance when searching for the rho value that yields value[int(n/2)]=0.\n'set_rho' must be an expression whose 'truthness' can be evaluated. If set_rho is True, the rho attribute in the python dp object will be set to the rho value obtained after iteration. If false, it will not be set.\n'set_bounds' must be an expression whose 'truthness' can be evaluated. If set_bounds is True, the python DecisionPolicy object's ´bounds´ attribute will be set to the upper and lower bounds in g space computed in the c++ instance. If false, it will do nothing."},
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
