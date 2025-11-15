#!/bin/bash

echo "Starting all jobs..."
sbatch job221.sh
sbatch job231.sh
sbatch job24.sh
echo "All jobs submitted."
sleep 2
clear
echo "Starting monitoring..."
sleep 3
clear
./monitor.sh
