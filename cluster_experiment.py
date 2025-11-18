from scalability_experiment import ClusterExperimentRunner, \
MRTreeModelTrainer, IntrusionDataloader, IntrusionPreprocessor
import argparse
import json
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cluster_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

current_path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Run cluster experiment')
parser.add_argument('-n', '--nodes', type=int,
                    help='Number of nodes to test', default=2)
args = parser.parse_args()

logger.info(f"Starting cluster experiment with {args.nodes} nodes")

SEED = 42
logger.info(f"Initializing experiment runner with seed {SEED}")
runner = ClusterExperimentRunner(IntrusionDataloader, IntrusionPreprocessor,
                                 MRTreeModelTrainer, seed=SEED)

logger.info("Running experiment...")
runner.run(args.nodes, K=[1], full_dataset=True)

if os.environ.get('SLURM_JOB_ID'):
    save_path = os.path.join(current_path, f'slurm_cluster_experiment_intrusion_mrtree_{args.nodes}_nodes_job{os.environ.get("SLURM_JOB_ID")}.json')
else:
    save_path = os.path.join(current_path, f'cluster_experiment_instrusion_mrtree_{args.nodes}_nodes.json')
logger.info(f"Saving metrics to {save_path}")
runner.save_trace(save_path)

print("="*40)
print(f"Résumé des métriques de l'expérience sur le cluster avec {args.nodes} nœuds :")
metrics = runner.save_trace()
print(json.dumps(metrics, indent=4))
print("="*40)

logger.info("Cluster experiment completed successfully")
