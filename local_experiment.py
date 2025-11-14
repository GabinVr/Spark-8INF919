from scalability_experiment import LocalExperimentRunner, AdultDataLoader, AdultDataPreprocessor, MRTreeModelTrainer

SEED = 42
runner = LocalExperimentRunner(AdultDataLoader, AdultDataPreprocessor, MRTreeModelTrainer, seed=SEED)
runner.run(K=[1, 8, 16, 32, 64])
runner.save_trace("local_experiment_trace.json")
runner.save_plot()