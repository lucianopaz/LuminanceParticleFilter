#!/bin/bash
# This is the job dispatcher called from slurm with the total number of
# jobs and the job index
if [ -z "$1" ]
  then
    echo "No task id supplied"
    exit 1
  else
    taskId=$1
fi
if [ -z "$2" ]
  then
    echo "Did not supply the number of processes"
    exit 2
  else
    numProcs=$2
fi

methods=("full_confidence" "confidence_only") # full, full_confidence, confidence_only, binary_confidence_only or full_binary_confidence
optimizer="cma" # cma, basinhopping or scipy.optimize.minimize or scipy.optimize.minimize_scalar methods
units="seconds" # seconds or milliseconds
plot_handler_rt_cutoff="6"
experiment="all" # all, luminancia, 2afc or auditivo
high_confidence_mapping_method="belief"
binary_split_method="median"
fixed_parameters=$'\'{"cost":null,"internal_var":null,"phase_out_prob":null,"dead_time":null,"dead_time_sigma":null}\''
optimizer_kwargs=$'\'{"restarts":2}\''
start_point_from_fit_output=$'\'{"method":"full","optimizer":"cma","suffix":"","cmapmeth":"log_odds"}\''

if [ $experiment == "all" ]
  then
    batch_size=176
elif [ $experiment == "luminancia" ]
  then
    batch_size=44
elif [ $experiment == "2afc" ]
  then
    batch_size=66
elif [ $experiment == "auditivo" ]
  then
    batch_size=66
else
    echo "Unknown experiment: $experiment"
    exit 3
fi
batch_index=$((($taskId-1)/$batch_size))
task=$((($taskId-1)%$batch_size+1))
ntasks=$(($numProcs-$batch_size*batch_index))
if (( $ntasks > $batch_size ))
  then
    ntasks=$batch_size
fi
method=${methods[$batch_index]}
echo "python fits_cognition.py -t $task -nt $ntasks -m $method -e $experiment \
--optimizer $optimizer --units $units --plot_handler_rt_cutoff $plot_handler_rt_cutoff \
--high_confidence_mapping_method $high_confidence_mapping_method \
--binary_split_method $binary_split_method \
--fixed_parameters $fixed_parameters \
--optimizer_kwargs $optimizer_kwargs \
--start_point_from_fit_output $start_point_from_fit_output"

#~ '-t' or '--task': Integer that identifies the task number when running multiple tasks
                   #~ in parallel. By default it is one based but this behavior can be
                   #~ changed with the option --task_base. [Default 1]
 #~ '-nt' or '--ntasks': Integer that identifies the number tasks working in parallel [Default 1]
 #~ '--save_plot_handler': This flag takes no value. If present, the plot_handler is saved.
 #~ '-v' or '--verbose': Activates info messages (by default only warnings and errors
                      #~ are printed). 
exit 0
