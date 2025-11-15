from scalability_experiment import ClusterExperimentRunner, \
MRTreeModelTrainer, AdultDataLoader, AdultDataPreprocessor
import argparse
import json
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cluster_experiment1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

current_path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Run cluster experiment')
parser.add_argument('-n', '--nodes', type=int,
                    help='Number of nodes to test', default=2)
parser.add_argument('-k', '--k_values', type=int, nargs='+',
                    help='List of K values for scale factor', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
args = parser.parse_args()

logger.info(f"Starting cluster experiment with {args.nodes} nodes and K values: {args.k_values}")

SEED = 42
logger.info(f"Initializing experiment runner with seed {SEED}")
runner = ClusterExperimentRunner(AdultDataLoader, AdultDataPreprocessor,
                                 MRTreeModelTrainer, seed=SEED)

logger.info("Running experiment...")
runner.run(args.nodes, K=args.k_values)

save_path = os.path.join(current_path, f'cluster_experiment_adult_mrtree_{args.nodes}_nodes.json')
logger.info(f"Saving metrics to {save_path}")
runner.save_metrics(save_path)

print("="*40)
print(f"Résumé des métriques de l'expérience sur le cluster avec {args.nodes} nœuds :")
metrics = runner.get_metrics()
print(json.dumps(metrics, indent=4))
print("="*40)

logger.info("Cluster experiment completed successfully")
