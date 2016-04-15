#include <Python.h>

#define CUSTOM_NPY_1_7_API_VERSION 7
#if NPY_API_VERSION<CUSTOM_NPY_1_7_API_VERSION
#define PyArray_SHAPE PyArray_DIMS
#endif
#undef CUSTOM_NPY_1_7_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "DecisionPolicy.hpp"

#include <cstdio>

DecisionPolicyDescriptor* get_descriptor(PyObject* py_dp){
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
	
	
	Py_DECREF(py_model_var);
	Py_DECREF(py_prior_mu_mean);
	Py_DECREF(py_prior_mu_var);
	Py_DECREF(py_n);
	Py_DECREF(py_dt);
	Py_DECREF(py_T);
	Py_DECREF(py_reward);
	Py_DECREF(py_penalty);
	Py_DECREF(py_iti);
	Py_DECREF(py_tp);
	Py_DECREF(py_cost);
	
	return new DecisionPolicyDescriptor(model_var, prior_mu_mean, prior_mu_var,
				   n, dt, T, reward, penalty, iti, tp, cost);
}

/* method xbounds(decisionPolicy, tolerance=1e-12, set_rho=True, set_bounds=True, return_values=False, root_bounds=None) */
static PyObject* dpmod_xbounds(PyObject* self, PyObject* args, PyObject* keywds){
	double tolerance = 1e-12;
	PyObject* py_dp;
	PyObject* py_set_rho_in_py_dp = Py_True;
	PyObject* py_touch_py_bounds = Py_True;
	PyObject* py_ret_values = Py_False;
	PyObject* py_root_bounds = Py_None;
	int set_rho_in_py_dp = 0;
	int touch_py_bounds = 0;
	int ret_values = 0;
	int must_dec_ref_py_bounds = 1;
	bool must_create_py_bounds = false;
	bool use_root_bounds = false;
	double lower_bound, upper_bound;
	PyObject* py_bounds = NULL;
	PyObject* py_out = NULL;
	PyObject* py_xub = NULL;
	PyObject* py_xlb = NULL;
	PyObject* py_value = NULL;
	PyObject* py_v_explore = NULL;
	PyObject* py_v1 = NULL;
	PyObject* py_v2 = NULL;
	DecisionPolicy* dp;
	DecisionPolicyDescriptor* dpd;
	
	
	static char* kwlist[] = {"decPol", "tolerance","set_rho","set_bounds","return_values","root_bounds", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|dOOOO", kwlist, &py_dp, &tolerance, &py_set_rho_in_py_dp, &py_touch_py_bounds, &py_ret_values, &py_root_bounds))
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
	ret_values = PyObject_IsTrue(py_ret_values);
	if (ret_values==-1) { // Failed to evaluate truth statement
		PyErr_SetString(PyExc_ValueError, "return_values needs to evaluate to a valid truth statemente");
		return NULL;
	}
	if (py_root_bounds!=Py_None){
		use_root_bounds = true;
		if (!PyArg_ParseTuple(py_root_bounds,"dd",&lower_bound,&upper_bound)){
			if (PyErr_ExceptionMatches(PyExc_SystemError)){
				PyErr_SetString(PyExc_TypeError,"Supplied parameter 'root_bounds' must be None or a tuple with two elements. Both elements must be floats");
			} else if (PyErr_ExceptionMatches(PyExc_TypeError)){
				PyErr_SetString(PyExc_TypeError,"Supplied parameter 'root_bounds' must be None or a tuple with two elements. Both elements must be floats");
			}
			return NULL;
		}
	}
	
	dpd = get_descriptor(py_dp);
	if (dpd==NULL){
		// An error occurred while getting the descriptor and the error message was set within get_descriptor
		return NULL;
	}
	int nT = (int)(dpd->T/dpd->dt)+1;
	npy_intp py_nT[1] = {nT};
	
	if (!touch_py_bounds){
		dp = new DecisionPolicy(*dpd);
	} else {
		npy_intp bounds_shape[2] = {2,nT};
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
				if (bounds_shape[i]!=PyArray_SHAPE((PyArrayObject*)py_bounds)[i]){
					// Attribute 'bounds' in DecisionPolicy instance does not have the correct shape. We must re create py_bounds
					Py_DECREF(py_bounds);
					must_create_py_bounds = true;
					break;
				}
			}
		}
		if (must_create_py_bounds){
			py_bounds = PyArray_SimpleNew(2,bounds_shape,NPY_DOUBLE);
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
		dp = new DecisionPolicy(*dpd,
								(double*)PyArray_GETPTR2((PyArrayObject*)py_bounds,(npy_intp)0,(npy_intp)0),
								(double*)PyArray_GETPTR2((PyArrayObject*)py_bounds,(npy_intp)1,(npy_intp)0),
								((int) PyArray_STRIDE((PyArrayObject*)py_bounds,1))/sizeof(double)); // We divide by sizeof(double) because strides determines the number of bytes to stride in each dimension. As we cast the supplied void pointer to double*, each element has sizeof(double) bytes instead of 1 byte.
	}
	
	if (!ret_values){
		py_out = PyTuple_New(2);
		py_xub = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_xlb = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		if (py_out==NULL || py_xub==NULL || py_xlb==NULL){
			PyErr_SetString(PyExc_MemoryError, "Out of memory");
			delete(dp);
			goto error_cleanup;
		}
		PyTuple_SET_ITEM(py_out, 0, py_xub); // Steals a reference to py_xub so no dec_ref must be called on py_xub on cleanup
		PyTuple_SET_ITEM(py_out, 1, py_xlb); // Steals a reference to py_xlb so no dec_ref must be called on py_xlb on cleanup
	} else {
		npy_intp py_value_shape[2] = {dp->nT,dp->n};
		npy_intp py_v_explore_shape[2] = {dp->nT-1,dp->n};
		npy_intp py_v12_shape[1] = {dp->n};
		
		py_out = PyTuple_New(6);
		py_xub = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_xlb = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_value = PyArray_SimpleNew(2, py_value_shape, NPY_DOUBLE);
		py_v_explore = PyArray_SimpleNew(2, py_v_explore_shape, NPY_DOUBLE);
		py_v1 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
		py_v2 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
		if (py_out==NULL || py_xub==NULL || py_xlb==NULL || py_value==NULL || py_v_explore==NULL || py_v1==NULL || py_v2==NULL){
			PyErr_SetString(PyExc_MemoryError, "Out of memory");
			delete(dp);
			goto error_cleanup;
		}
		PyTuple_SET_ITEM(py_out, 0, py_xub); // Steals a reference to py_xub so no dec_ref must be called on py_xub on cleanup
		PyTuple_SET_ITEM(py_out, 1, py_xlb); // Steals a reference to py_xlb so no dec_ref must be called on py_xlb on cleanup
		PyTuple_SET_ITEM(py_out, 2, py_value); // Steals a reference to py_value so no dec_ref must be called on py_value on cleanup
		PyTuple_SET_ITEM(py_out, 3, py_v_explore); // Steals a reference to py_v_explore so no dec_ref must be called on py_v_explore on cleanup
		PyTuple_SET_ITEM(py_out, 4, py_v1); // Steals a reference to py_v1 so no dec_ref must be called on py_v1 on cleanup
		PyTuple_SET_ITEM(py_out, 5, py_v2); // Steals a reference to py_v2 so no dec_ref must be called on py_v2 on cleanup
	}
	
	if (use_root_bounds){
		dp->iterate_rho_value(tolerance,lower_bound,upper_bound);
	} else {
		dp->iterate_rho_value(tolerance);
	}
	if (set_rho_in_py_dp){
		if (PyObject_SetAttrString(py_dp,"rho",Py_BuildValue("d",dp->rho))==-1){
			PyErr_SetString(PyExc_ValueError, "Could not set decisionPolicy property rho");
			delete(dp);
			goto error_cleanup;
		}
	}
	// Backpropagate and compute bounds in the diffusion space
	if (!ret_values) {
		dp->backpropagate_value();
	} else {
		dp->backpropagate_value(dp->rho,true,
				(double*) PyArray_DATA((PyArrayObject*) py_value),
				(double*) PyArray_DATA((PyArrayObject*) py_v_explore),
				(double*) PyArray_DATA((PyArrayObject*) py_v1),
				(double*) PyArray_DATA((PyArrayObject*) py_v2));
	}
	dp->x_ubound((double*) PyArray_DATA((PyArrayObject*) py_xub));
	dp->x_lbound((double*) PyArray_DATA((PyArrayObject*) py_xlb));
	
	// normal_cleanup
	delete(dpd);
	delete(dp);
	if (must_dec_ref_py_bounds) Py_XDECREF(py_bounds);
	return py_out;

error_cleanup:
	delete(dpd);
	if (must_dec_ref_py_bounds) Py_XDECREF(py_bounds);
	Py_XDECREF(py_xub);
	Py_XDECREF(py_xlb);
	Py_XDECREF(py_value);
	Py_XDECREF(py_v_explore);
	Py_XDECREF(py_v1);
	Py_XDECREF(py_v2);
	Py_XDECREF(py_out);
	return NULL;
}

/* method xbounds_fixed_rho(decisionPolicy, rho=None, set_bounds=False, return_values=False) */
static PyObject* dpmod_xbounds_fixed_rho(PyObject* self, PyObject* args, PyObject* keywds){
	PyObject* py_dp;
	PyObject* py_touch_py_bounds = Py_False;
	PyObject* py_ret_values = Py_False;
	PyObject* py_rho = Py_None;
	double rho;
	int touch_py_bounds = 0;
	int ret_values = 0;
	int must_dec_ref_py_bounds = 1;
	bool must_create_py_bounds = false;
	PyObject* py_bounds = NULL;
	PyObject* py_out = NULL;
	PyObject* py_xub = NULL;
	PyObject* py_xlb = NULL;
	PyObject* py_value = NULL;
	PyObject* py_v_explore = NULL;
	PyObject* py_v1 = NULL;
	PyObject* py_v2 = NULL;
	DecisionPolicy* dp;
	DecisionPolicyDescriptor* dpd;
	
	
	static char* kwlist[] = {"decPol","rho","set_bounds","return_values", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|OOO", kwlist, &py_dp, &py_rho, &py_touch_py_bounds, &py_ret_values))
		return NULL;
	
	if (py_rho==Py_None) { // Use dp rho value
		py_rho = PyObject_GetAttrString(py_dp,"rho");
		if (py_rho==NULL){
			PyErr_SetString(PyExc_ValueError, "DecisionPolicy instance has no rho attribute. You should set it or pass rho as the second parameter to the 'value' function");
			return NULL;
		}
		rho = PyFloat_AsDouble(py_rho);
		if (PyErr_Occurred()!=NULL){
			return NULL;
		}
	} else {
		rho = PyFloat_AsDouble(py_rho);
		if (PyErr_Occurred()!=NULL){
			return NULL;
		}
	}
	touch_py_bounds = PyObject_IsTrue(py_touch_py_bounds);
	if (touch_py_bounds==-1) { // Failed to evaluate truth statement
		PyErr_SetString(PyExc_ValueError, "set_bounds needs to evaluate to a valid truth statemente");
		return NULL;
	}
	ret_values = PyObject_IsTrue(py_ret_values);
	if (ret_values==-1) { // Failed to evaluate truth statement
		PyErr_SetString(PyExc_ValueError, "return_values needs to evaluate to a valid truth statemente");
		return NULL;
	}
	
	dpd = get_descriptor(py_dp);
	if (dpd==NULL){
		// An error occurred while getting the descriptor and the error message was set within get_descriptor
		return NULL;
	}
	int nT = (int)(dpd->T/dpd->dt)+1;
	npy_intp py_nT[1] = {nT};
	
	if (!touch_py_bounds){
		dp = new DecisionPolicy(*dpd);
	} else {
		npy_intp bounds_shape[2] = {2,nT};
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
				if (bounds_shape[i]!=PyArray_SHAPE((PyArrayObject*)py_bounds)[i]){
					// Attribute 'bounds' in DecisionPolicy instance does not have the correct shape. We must re create py_bounds
					Py_DECREF(py_bounds);
					must_create_py_bounds = true;
					break;
				}
			}
		}
		if (must_create_py_bounds){
			py_bounds = PyArray_SimpleNew(2,bounds_shape,NPY_DOUBLE);
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
		dp = new DecisionPolicy(*dpd,
								(double*)PyArray_GETPTR2((PyArrayObject*)py_bounds,(npy_intp)0,(npy_intp)0),
								(double*)PyArray_GETPTR2((PyArrayObject*)py_bounds,(npy_intp)1,(npy_intp)0),
								((int) PyArray_STRIDE((PyArrayObject*)py_bounds,1))/sizeof(double)); // We divide by sizeof(double) because strides determines the number of bytes to stride in each dimension. As we cast the supplied void pointer to double*, each element has sizeof(double) bytes instead of 1 byte.
	}
	dp->rho = rho;
	
	if (!ret_values){
		py_out = PyTuple_New(2);
		py_xub = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_xlb = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		if (py_out==NULL || py_xub==NULL || py_xlb==NULL){
			PyErr_SetString(PyExc_MemoryError, "Out of memory");
			delete(dp);
			goto error_cleanup;
		}
		PyTuple_SET_ITEM(py_out, 0, py_xub); // Steals a reference to py_xub so no dec_ref must be called on py_xub on cleanup
		PyTuple_SET_ITEM(py_out, 1, py_xlb); // Steals a reference to py_xlb so no dec_ref must be called on py_xlb on cleanup
	} else {
		npy_intp py_value_shape[2] = {dp->nT,dp->n};
		npy_intp py_v_explore_shape[2] = {dp->nT-1,dp->n};
		npy_intp py_v12_shape[1] = {dp->n};
		
		py_out = PyTuple_New(6);
		py_xub = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_xlb = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_value = PyArray_SimpleNew(2, py_value_shape, NPY_DOUBLE);
		py_v_explore = PyArray_SimpleNew(2, py_v_explore_shape, NPY_DOUBLE);
		py_v1 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
		py_v2 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
		if (py_out==NULL || py_xub==NULL || py_xlb==NULL || py_value==NULL || py_v_explore==NULL || py_v1==NULL || py_v2==NULL){
			PyErr_SetString(PyExc_MemoryError, "Out of memory");
			delete(dp);
			goto error_cleanup;
		}
		PyTuple_SET_ITEM(py_out, 0, py_xub); // Steals a reference to py_xub so no dec_ref must be called on py_xub on cleanup
		PyTuple_SET_ITEM(py_out, 1, py_xlb); // Steals a reference to py_xlb so no dec_ref must be called on py_xlb on cleanup
		PyTuple_SET_ITEM(py_out, 2, py_value); // Steals a reference to py_value so no dec_ref must be called on py_value on cleanup
		PyTuple_SET_ITEM(py_out, 3, py_v_explore); // Steals a reference to py_v_explore so no dec_ref must be called on py_v_explore on cleanup
		PyTuple_SET_ITEM(py_out, 4, py_v1); // Steals a reference to py_v1 so no dec_ref must be called on py_v1 on cleanup
		PyTuple_SET_ITEM(py_out, 5, py_v2); // Steals a reference to py_v2 so no dec_ref must be called on py_v2 on cleanup
	}
	
	// Backpropagate and compute bounds in the diffusion space
	if (!ret_values) {
		dp->backpropagate_value(dp->rho,true);
	} else {
		dp->backpropagate_value(dp->rho,true,
				(double*) PyArray_DATA((PyArrayObject*) py_value),
				(double*) PyArray_DATA((PyArrayObject*) py_v_explore),
				(double*) PyArray_DATA((PyArrayObject*) py_v1),
				(double*) PyArray_DATA((PyArrayObject*) py_v2));
	}
	dp->x_ubound((double*) PyArray_DATA((PyArrayObject*) py_xub));
	dp->x_lbound((double*) PyArray_DATA((PyArrayObject*) py_xlb));
	
	// normal_cleanup
	delete(dpd);
	delete(dp);
	if (must_dec_ref_py_bounds) Py_XDECREF(py_bounds);
	return py_out;

error_cleanup:
	delete(dpd);
	if (must_dec_ref_py_bounds) Py_XDECREF(py_bounds);
	Py_XDECREF(py_xub);
	Py_XDECREF(py_xlb);
	Py_XDECREF(py_value);
	Py_XDECREF(py_v_explore);
	Py_XDECREF(py_v1);
	Py_XDECREF(py_v2);
	Py_XDECREF(py_out);
	return NULL;
}

/* method values(decisionPolicy, rho=None) */
static PyObject* dpmod_values(PyObject* self, PyObject* args, PyObject* keywds){
	PyObject* py_dp;
	PyObject* py_rho = Py_None;
	double rho;
	PyObject* py_out = NULL;
	PyObject* py_value = NULL;
	PyObject* py_v_explore = NULL;
	PyObject* py_v1 = NULL;
	PyObject* py_v2 = NULL;
	DecisionPolicyDescriptor* dpd;
	
	static char* kwlist[] = {"decPol", "rho", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|O", kwlist, &py_dp, &py_rho))
		return NULL;
	
	if (py_rho==Py_None) { // Use dp rho value
		py_rho = PyObject_GetAttrString(py_dp,"rho");
		if (py_rho==NULL){
			PyErr_SetString(PyExc_ValueError, "DecisionPolicy instance has no rho attribute. You should set it or pass rho as the second parameter to the 'value' function");
			return NULL;
		}
		rho = PyFloat_AsDouble(py_rho);
		if (PyErr_Occurred()!=NULL){
			return NULL;
		}
	} else {
		rho = PyFloat_AsDouble(py_rho);
		if (PyErr_Occurred()!=NULL){
			return NULL;
		}
	}
	
	dpd = get_descriptor(py_dp);
	if (dpd==NULL){
		// An error occurred while getting the descriptor and the error message was set within get_descriptor
		return NULL;
	}
	
	DecisionPolicy dp = DecisionPolicy(*dpd);
	dp.rho = rho;
	
	npy_intp py_value_shape[2] = {dp.nT,dp.n};
	npy_intp py_v_explore_shape[2] = {dp.nT-1,dp.n};
	npy_intp py_v12_shape[1] = {dp.n};
	
	py_out = PyTuple_New(4);
	py_value = PyArray_SimpleNew(2, py_value_shape, NPY_DOUBLE);
	py_v_explore = PyArray_SimpleNew(2, py_v_explore_shape, NPY_DOUBLE);
	py_v1 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
	py_v2 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
	
	if (py_out==NULL || py_value==NULL || py_v_explore==NULL || py_v1==NULL || py_v2==NULL){
		PyErr_SetString(PyExc_MemoryError, "Out of memory");
		goto error_cleanup;
	}
	PyTuple_SET_ITEM(py_out, 0, py_value); // Steals a reference to py_value so no dec_ref must be called on py_value on cleanup
	PyTuple_SET_ITEM(py_out, 1, py_v_explore); // Steals a reference to py_v_explore so no dec_ref must be called on py_v_explore on cleanup
	PyTuple_SET_ITEM(py_out, 2, py_v1); // Steals a reference to py_v1 so no dec_ref must be called on py_v1 on cleanup
	PyTuple_SET_ITEM(py_out, 3, py_v2); // Steals a reference to py_v2 so no dec_ref must be called on py_v2 on cleanup
		
	dp.backpropagate_value(dp.rho,false,
			(double*) PyArray_DATA((PyArrayObject*) py_value),
			(double*) PyArray_DATA((PyArrayObject*) py_v_explore),
			(double*) PyArray_DATA((PyArrayObject*) py_v1),
			(double*) PyArray_DATA((PyArrayObject*) py_v2));
	
	delete(dpd);
	return py_out;

error_cleanup:
	delete(dpd);
	Py_XDECREF(py_value);
	Py_XDECREF(py_v_explore);
	Py_XDECREF(py_v1);
	Py_XDECREF(py_v2);
	Py_XDECREF(py_out);
	return NULL;
}

/* method rt(decisionPolicy, mu, bounds=None) */
static PyObject* dpmod_rt(PyObject* self, PyObject* args, PyObject* keywds){
	PyObject* py_dp;
	PyObject* py_bounds = Py_None;
	double mu, rho;
	bool must_decref_py_bounds = false;
	PyArrayObject* py_xub, *py_xlb;
	
	PyObject* py_out = NULL;
	PyObject* py_g1 = NULL;
	PyObject* py_g2 = NULL;
	DecisionPolicyDescriptor* dpd;
	
	static char* kwlist[] = {"decPol", "mu", "bounds", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "Od|O", kwlist, &py_dp, &mu, &py_bounds))
		return NULL;
	
	dpd = get_descriptor(py_dp);
	if (dpd==NULL){
		// An error occurred while getting the descriptor and the error message was set within get_descriptor
		return NULL;
	}
	npy_intp py_nT[1] = {dpd->nT};
	
	PyObject* py_rho = PyObject_GetAttrString(py_dp,"rho");
	if (py_rho==NULL){
		PyErr_SetString(PyExc_ValueError, "decisionPolicy instance has no rho value set");
		return NULL;
	}
	rho = PyFloat_AsDouble(py_rho);
	if (PyErr_Occurred()!=NULL){
		Py_XDECREF(py_rho);
		return NULL;
	}
	Py_DECREF(py_rho);
	
	DecisionPolicy dp = DecisionPolicy(*dpd);
	dp.rho = rho;
	
	if (py_bounds==Py_None) { // Compute xbounds if they are not provided
		PyObject* args2 = PyTuple_Pack(1,py_dp);
		py_bounds = dpmod_xbounds(self,args2,NULL);
		Py_DECREF(args2);
		if (py_bounds==NULL){
			return NULL;
		}
		must_decref_py_bounds = true;
	}
	
	if (!PyArg_ParseTuple(py_bounds,"O!O!", &PyArray_Type, &py_xub, &PyArray_Type, &py_xlb))
		goto error_cleanup;
	if (PyArray_NDIM(py_xub)!=1 || PyArray_NDIM(py_xlb)!=1){
		// Attribute 'bounds' in DecisionPolicy instance does not have the correct shape. We must re create py_bounds
		PyErr_SetString(PyExc_ValueError,"Supplied bounds must be numpy arrays with one dimension");
		goto error_cleanup;
	} else if (PyArray_SHAPE(py_xub)[0]!=py_nT[0] || PyArray_SHAPE(py_xlb)[0]!=py_nT[0]) {
		PyErr_Format(PyExc_ValueError,"Supplied bounds must be numpy arrays of shape (%d)",dpd->nT);
		goto error_cleanup;
	}
	
	py_out = PyTuple_New(2);
	py_g1 = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
	py_g2 = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
	
	if (py_out==NULL || py_g1==NULL || py_g2==NULL){
		PyErr_SetString(PyExc_MemoryError, "Out of memory");
		goto error_cleanup;
	}
	PyTuple_SET_ITEM(py_out, 0, py_g1); // Steals a reference
	PyTuple_SET_ITEM(py_out, 1, py_g2); // Steals a reference
		
	dp.rt(mu,(double*) PyArray_DATA((PyArrayObject*) py_g1),
			 (double*) PyArray_DATA((PyArrayObject*) py_g2),
			 (double*) PyArray_DATA((PyArrayObject*) py_xub),
			 (double*) PyArray_DATA((PyArrayObject*) py_xlb));
	
	delete(dpd);
	if (must_decref_py_bounds){
		Py_XDECREF(py_bounds);
		// The elements in py_bounds are also decref'ed so only calling
		// decref on the py_bounds instance is sufficient
	}
	return py_out;

error_cleanup:
	delete(dpd);
	if (must_decref_py_bounds){
		Py_XDECREF(py_bounds);
		// The elements in py_bounds are also decref'ed so only calling
		// decref on the py_bounds instance is sufficient
	}
	Py_XDECREF(py_g1);
	Py_XDECREF(py_g2);
	Py_XDECREF(py_out);
	return NULL;
}

static PyMethodDef DPMethods[] = {
    {"xbounds", (PyCFunction) dpmod_xbounds, METH_VARARGS | METH_KEYWORDS,
     "Computes the decision bounds in x(t) space (i.e. the accumulated sensory input space)\n\n  (xub, xlb) = xbounds(dp, tolerance=1e-12, set_rho=False, set_bounds=False, return_values=False, root_bounds=None)\n\n(xub, xlb, value, v_explore, v1, v2) = xbounds(dp, ..., return_values=True)\n\nComputes the decision bounds for a decisionPolicy instance specified in 'dp'.\nThis function is more memory and computationally efficient than calling dp.invert_belief();dp.value_dp(); xb = dp.belief_bound_to_x_bound(b); from python. Another difference is that this function returns a tuple of (upper_bound, lower_bound) instead of a numpy array whose first element is upper_bound and second element is lower_bound.\n'tolerance' is a float that indicates the tolerance when searching for the rho value that yields value[int(n/2)]=0.\n'set_rho' must be an expression whose 'truthness' can be evaluated. If set_rho is True, the rho attribute in the python dp object will be set to the rho value obtained after iteration. If false, it will not be set.\n'set_bounds' must be an expression whose 'truthness' can be evaluated. If set_bounds is True, the python DecisionPolicy object's ´bounds´ attribute will be set to the upper and lower bounds in g space computed in the c++ instance. If false, it will do nothing.\nIf 'return_values' evaluates to True, then the function returns four extra numpy arrays: value, v_explore, v1 and v2. 'value' is an nT by n shaped array that holds the value of a given g at time t. 'v_explore' has shape nT-1 by n and holds the value of exploring at time t with a given g. v1 and v2 are values of immediately deciding for option 1 or 2, and are one dimensional arrays with n elements.\n'root_bounds' must be a tuple of two elements: (lower_bound, upper_bound). Both 'lower_bound' and 'upper_bound' must be floats that represent the lower and upper bounds in which to perform the root finding of rho."},
    {"xbounds_fixed_rho", (PyCFunction) dpmod_xbounds_fixed_rho, METH_VARARGS | METH_KEYWORDS,
     "Computes the decision bounds in x(t) space (i.e. the accumulated sensory input space) without iterating the value of rho\n\n  (xub, xlb) = xbounds_fixed_rho(dp, rho=None, set_bounds=False, return_values=False)\n\n(xub, xlb, value, v_explore, v1, v2) = xbounds_fixed_rho(dp, ..., return_values=True)\n\nComputes the decision bounds for a decisionPolicy instance specified in 'dp' for a given rho value.\nThis function is more memory and computationally efficient than calling dp.invert_belief();dp.value_dp(); xb = dp.belief_bound_to_x_bound(b); from python. Another difference is that this function returns a tuple of (upper_bound, lower_bound) instead of a numpy array whose first element is upper_bound and second element is lower_bound.\n'rho' is the fixed reward rate value used to compute the decision bounds and values. If rho=None, then the DecisionPolicy instance's rho is used.\n'set_bounds' must be an expression whose 'truthness' can be evaluated. If set_bounds is True, the python DecisionPolicy object's ´bounds´ attribute will be set to the upper and lower bounds in g space computed in the c++ instance. If false, it will do nothing.\nIf 'return_values' evaluates to True, then the function returns four extra numpy arrays: value, v_explore, v1 and v2. 'value' is an nT by n shaped array that holds the value of a given g at time t. 'v_explore' has shape nT-1 by n and holds the value of exploring at time t with a given g. v1 and v2 are values of immediately deciding for option 1 or 2, and are one dimensional arrays with n elements."},
    {"values", (PyCFunction) dpmod_values, METH_VARARGS | METH_KEYWORDS,
     "Computes the values for a given reward rate, rho, and decisionPolicy parameters.\n\n(value, v_explore, v1, v2) = values(dp,rho=None)\n\nComputes the value for a given belief g as a function of time for a supplied reward rate, rho. If rho is set to None, then the decisionPolicy instance's rho attribute will be used.\nThis function is more memory and computationally efficient than calling dp.invert_belief();dp.value_dp(); from python. The function returns a tuple of four numpy arrays: value, v_explore, v1 and v2. 'value' is an nT by n shaped array that holds the value of a given g at time t. 'v_explore' has shape nT-1 by n and holds the value of exploring at time t with a given g. v1 and v2 are values of immediately deciding for option 1 or 2, and are one dimensional arrays with n elements."},
    {"rt", (PyCFunction) dpmod_rt, METH_VARARGS | METH_KEYWORDS,
     "Computes the rt distribution for a given drift rate, mu, decisionPolicy parameters and decision bounds in x space, bounds.\n\n(g1, g2) = values(dp,mu,bounds=None)\n\nComputes the rt distribution for selecting options 1 or 2 for a given drift rate mu and the parameters in the decisionPolicy instance dp. If bounds is None, then xbounds is called with default parameters to compute the decision bounds in x space. To avoid this, supply a tuple (xub,xlb) as the one that is returned by the function xbounds. xub and xlb must be one dimensional numpy arrays with the same elements as dp.t."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
    /* module initialisation for Python 3 */
    static struct PyModuleDef dpmodule = {
       PyModuleDef_HEAD_INIT,
       "dp",   /* name of module */
       "Module to compute the decision bounds and values for bayesian inference",
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
