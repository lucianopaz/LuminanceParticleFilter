#!/bin/bash
# This is the job dispatcher called from slurm with the total number of
# jobs and the job index
read -r -d '' usage <<- EOM
	$(basename "$0") help script
	This bash script works as an interface between the cluster's batch
	script used to supply jobs with the grid engine, and the python
	script fits_cognition.py that is used for data fitting.

	Syntax:
	dispatcher task_id number_of_processes

	Input:
	The script takes two mandatory inputs
	task_id: An integer that identifies the cluster's task index that
	    will be runned. The task_id must be 1 or higher, i.e. the task
	    base is one as opposed to the zero based MPI task index.
	number_of_processes: An integer that specifies the number of tasks
	    that will run in parallel. Be aware that the processes may run
	    at different times, and not necessarily all at the same time.
	    If the batch script is supplied as a job array, this should be
	    equal to the array length. If the script is supplied as a mpirun
	    or mpiexec, it should be equal to the number of tasks in the
	    MPI environment.

	Exit statuses:
	0: Success
	2: Missing task_id input
	3: Missing number_of_processes input
	4: Unknown experiment (current implementation has the experiment
	     hardcoded but future versions may allow for flexible assignment).
	Other exit statuses will be returned if the python script were to
	raise an exception and fail.
	 
EOM
if [ -z "$1" ]; then
	echo "No task id supplied"
	exit 2
else
	task_id=$1
	if [ $task_id == "-h" ] || [ $task_id == "--help" ]; then
		echo "$usage"
		exit 0
	fi
fi
if [ -z "$2" ]; then
	echo "Did not supply the number of processes"
	exit 3
else
	number_of_processes=$2
fi
if [ -z "$3" ]; then
	python="python2.7"
else
	python=$3
fi

methods=("full_confidence" "confidence_only") # full, full_confidence, confidence_only, binary_confidence_only or full_binary_confidence
optimizer="cma" # cma, basinhopping or scipy.optimize.minimize or scipy.optimize.minimize_scalar methods
units="seconds" # seconds or milliseconds
plot_handler_rt_cutoff="6"
experiment="all" # all, luminancia, 2afc or auditivo
high_confidence_mapping_method="belief" # log_odds or belief
binary_split_method="median" # median, half or mean
fixed_parameters='{"cost":null,"internal_var":null,"phase_out_prob":null,"dead_time":null,"dead_time_sigma":null}'
optimizer_kwargs='{"restarts":2}'
start_point_from_fit_output='{"method":"full","optimizer":"cma","suffix":"","cmapmeth":"log_odds"}'

if [ $experiment == "all" ]; then
	batch_size=176
elif [ $experiment == "luminancia" ]; then
	batch_size=44
elif [ $experiment == "2afc" ]; then
	batch_size=66
elif [ $experiment == "auditivo" ]; then
	batch_size=66
else
	echo "Unknown experiment: $experiment"
	exit 4
fi
available_batch_size=$(($number_of_processes/2))
if (( $task_id <= $available_batch_size )); then
	batch_index=0
	task=$task_id
	ntasks=$available_batch_size
else
	batch_index=1
	task=$(($task_id-$available_batch_size))
	ntasks=$(($number_of_processes-$available_batch_size))
fi
method=${methods[$batch_index]}
command="$python fits_cognition.py -t $task -nt $ntasks -m $method -e $experiment \
--optimizer $optimizer --units $units --plot_handler_rt_cutoff $plot_handler_rt_cutoff \
--high_confidence_mapping_method $high_confidence_mapping_method \
--binary_split_method $binary_split_method \
--fixed_parameters $fixed_parameters \
--optimizer_kwargs $optimizer_kwargs \
--start_point_from_fit_output $start_point_from_fit_output \
--save_plot_handler -v"
exit_status=eval $command
exit $exit_status
