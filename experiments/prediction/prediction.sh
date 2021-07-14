#!/bin/bash

dataset='../../data/data_full_id.csv'
arrayMax=('')
arrayRan=('' '--random')
arrayTrain=(0.2 0.5 0.8)
arrayTest=(0 1 2 3)

# Make outfile if it does not exist yet
outdir=result
mkdir -p $outdir

# Loop over all configurations
for max in "${arrayMax[@]}"; do
  for train in "${arrayTrain[@]}"; do
    for random in "${arrayRan[@]}"; do
      # Initialise PID list
      PID_LIST="";
      # Fill PID_LIST with parallel processes
      for test in "${arrayTest[@]}"; do
        # Set device
        device_number=$((test % 4))
        device=cuda:$device_number

        m=($max)
        r=${random/--/""}
        outfile=$outdir/max_${m[1]}_ratio_${train}_${r}_${test}.result

        # Set command
        cmd="python3 prediction_original.py $dataset $max --top 5 --ratio $train --batch-size 128 --device $device $random"

        echo "Process \"$cmd\" started";
        # Start process and capture PID
        $cmd > $outfile & pid=$!
        # Add PID to list
        PID_LIST+=" $pid";

        # Run experiment
        # echo python3 prediction_original.py $dataset $max --top 5 --ratio $train --batch-size 128 $random
      done
      # Wait until all PIDs are complete
      wait $PID_LIST
      echo "All processes have completed";
    done
  done
done

# Trap ctrl+c
trap "kill $PID_LIST" SIGINT
