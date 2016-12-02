#-*- coding: UTF-8 -*-
"""
cogpack __init__.py

This package was created to work as an interface to fit a dynamical
decision making model based on bayesian inference to the three 2AFC
tasks in Ais et al 2016 (Ais, J., Zylberberg, A., Barttfeld, P., and
Sigman, M. (2016). "Individual consistency in the accuracy and
distribution of confidence judgments". Cognition, 146, 377–386.
http://doi.org/10.1016/j.cognition.2015.10.006)

There are 5 main classes:
SubjectSession: Handles the experimental data loading
DecisionPolicy: The class that implements the theoretical model
Fitter: A class that works as an interface between the model, the
	experimental data, and performs parameter fits on them
Fitter_plot_handler: A class that is normally created by a Fitter
	instance. Serves as a plotting interface
Analyzer: A class that constructs summary statistics from a set of saved
	Fitter instances and the experimental data. Implements several
	analyzes on the parameters such as correlation analysis and
	parameter clustering.

Extra modules loaded:
data_io_cognition: Implements the SubjectSession class and several other
	relevant "static" methods.
cost_time: Implements the DecisionPolicy class and several other
	relevant "static" methods.
fits_cognition: Implements the Fitter and Fitter_plot_handler classes
	and some other "static" methods.
analysis: Implements the Analyzer class and several other relevant
	"static" methods.
utils: A utility module with auxiliary methods.
dp: A wrapper to c++ implementations specific to the DecisionPolicy
	class.
cma: A module that implements Covariance Matrix Adaptation Evolutionary
	Strategy optimization library

Convenience functions:
Fitter_filename: A function that can construct the formatted filename
	where a Fitter instance of the desired characteristics should be
	located
load_Fitter_from_file: A function that loads a Fitter instance from
	a file.

"""
from data_io_cognition import SubjectSession
from cost_time import DecisionPolicy
from fits_cognition import Fitter,Fitter_plot_handler,Fitter_filename,load_Fitter_from_file
from analysis import Analyzer
