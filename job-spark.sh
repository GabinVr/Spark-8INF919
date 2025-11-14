#!/bin/bash
#SBATCH --account=def-sponsor00
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --ntasks-per-node=1

module load StdEnv/2023
module load scipy-stack
module load spark/3.5.6



export MKL_NUM_THREADS=1
export SPARK_IDENT_STRING=$SLURM_JOBID
export SPARK_WORKER_DIR=$SLURM_TMPDIR
export SLURM_SPARK_MEM=$(printf "%.0f" $((${SLURM_MEM_PER_NODE} * 0.95)))

start-master.sh
sleep 5
MASTER_URL=$(grep -Po '(?=spark://).*' $SPARK_LOG_DIR/spark-$SPARK_IDENT_STRING-org.apache.spark.deploy.master*.out)

NWORKERS=$((SLURM_TASKS_PER_NODE * SLURM_JOB_NUM_NODES -1))
SPARK_NO_DAEMONIZE=1 \
  srun -n ${NWORKERS} \
       -N ${NWORKERS} \
       --label \
       --output=$SPARK_LOG_DIR/spark-%j-workers.out \
       start-slave.sh \
       -m ${SLURM_SPARK_MEM}M 

slaves_pids=$!

SLURM_SPARK_SUBMIT="srun -n 2 -N 2 spark-submit --master ${MASTER_URL} --executor-memory ${SLURM_SPARK_MEM}M cluster_experiment.py --nodes 2"
$SLURM_SPARK_SUBMIT
#$SLURM_SPARK_SUBMIT --class org.apache.spark.examples.SparkPi $SPARK_HOME/examples/jars/spark-examples_2.11-2.3.0.jar 1000
#$SLURM_SPARK_SUBMIT --class org.apache.spark.examples.SparkLR $SPARK_HOME/examples/jars/spark-examples_2.11-2.3.0.jar 1000
SLURM_SPARK_SUBMIT="srun -n 3 -N 3 spark-submit --master ${MASTER_URL} --executor-memory ${SLURM_SPARK_MEM}M cluster_experiment.py --nodes 3"
$SLURM_SPARK_SUBMIT
SLURM_SPARK_SUBMIT="srun -n 4 -N 4 spark-submit --master ${MASTER_URL} --executor-memory ${SLURM_SPARK_MEM}M cluster_experiment.py --nodes 4"
$SLURM_SPARK_SUBMIT
SLURM_SPARK_SUBMIT="srun -n 4 -N 4 spark-submit --master ${MASTER_URL} --executor-memory ${SLURM_SPARK_MEM}M baseline_method.py --nodes 4"
$SLURM_SPARK_SUBMIT
kill $slaves_pids
stop-master.sh