from scalability_experiment import ClusterExperimentRunner, \
MRTreeModelTrainer, IntrusionDataloader, IntrusionPreprocessor
import argparse
import json

parser = argparse.ArgumentParser(description='Run cluster experiment')
parser.add_argument('-n', '--nodes', type=int,
                    help='Number of nodes to test')
args = parser.parse_args()


SEED = 42
runner = ClusterExperimentRunner(IntrusionDataloader, IntrusionPreprocessor,
                                 MRTreeModelTrainer, seed=SEED)
runner.run(args.nodes, K=[1], full_dataset=True)
runner.save_plot(f'cluster_experiment_adult_mrtree_{args.nodes}_nodes.png')

print("="*40)
print(f"Résumé des métriques de l'expérience sur le cluster avec {args.nodes} nœuds :")
metrics = runner.get_metrics()
print(json.dumps(metrics, indent=4))
print("="*40)

