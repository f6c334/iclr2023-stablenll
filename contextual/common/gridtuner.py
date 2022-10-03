import itertools
import random

from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module


class GridSearchOracle(oracle_module.Oracle):

  def __init__(
      self,
      objective=None,
      max_trials=10,
      seed=None,
      hyperparameters=None,
      allow_new_entries=True,
      tune_new_entries=True,
  ):
    super(GridSearchOracle, self).__init__(
        objective=objective,
        max_trials=max_trials,
        hyperparameters=hyperparameters,
        tune_new_entries=tune_new_entries,
        allow_new_entries=allow_new_entries,
        seed=seed,
    )
    self.all_hyperparameter_combinations = None

  def populate_space(self, trial_id):
    if self.all_hyperparameter_combinations is None:
      self.generate_hyperparameter_combinations()
    
    trial_num = len(self.trials.items())

    if trial_num < len(self.all_hyperparameter_combinations):
      values = self.all_hyperparameter_combinations[trial_num].values
      values_hash = self._compute_values_hash(values)
      self._tried_so_far.add(values_hash)
      return {"status": trial_module.TrialStatus.RUNNING, "values": values}
    else:
      return {"status": trial_module.TrialStatus.STOPPED, "values": None}

  def generate_hyperparameter_combinations(self):
    self.all_hyperparameter_combinations = []
    for hpc in itertools.product(*[hp.values for hp in self.hyperparameters.space]):
      hps = hp_module.HyperParameters()
      for i, hp in enumerate(self.hyperparameters.space):
        hps.merge([hp])
        if hps.is_active(hp):  # Only active params in `values`.
          hps.values[hp.name] = hpc[i]
      self.all_hyperparameter_combinations.append(hps)
    random.shuffle(self.all_hyperparameter_combinations)


class GridSearch(tuner_module.Tuner):

  def __init__(self,
               hypermodel=None,
               objective=None,
               max_trials=10,
               seed=None,
               hyperparameters=None,
               tune_new_entries=True,
               allow_new_entries=True,
               **kwargs):
    self.seed = seed
    oracle = GridSearchOracle(
        objective=objective,
        max_trials=max_trials,
        seed=seed,
        hyperparameters=hyperparameters,
        tune_new_entries=tune_new_entries,
        allow_new_entries=allow_new_entries,
    )
    super(GridSearch, self).__init__(oracle, hypermodel, **kwargs)