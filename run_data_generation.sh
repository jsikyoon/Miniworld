#!/bin/bash

num_episodes=10000
num_processes=10

# Compute the number of episodes per process
episodes_per_process=$((num_episodes / num_processes))

for i in $(seq 0 $(($num_processes - 1)))
do
  # Compute the start and end index for the current process
  start_index=$((i * episodes_per_process))
  end_index=$((start_index + episodes_per_process - 1))

  # Launch the process with the start and end index arguments
  xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
    python -m miniworld_test.generate_data_for_stmem_3d_gen.py \
    $start_index $end_index &

  sleep 10
done

# Wait for all processes to finish
wait
