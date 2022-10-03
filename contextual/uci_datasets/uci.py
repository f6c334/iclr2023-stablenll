import io
import os

import pandas as pd

from scipy.io import arff



uci_folder = os.path.dirname(__file__)

###################################################################################################
### UCI DATASET GETTERS ###########################################################################
""" UCI Carbon Nanotubes Regression Set (10721, 5, 3)
    https://archive.ics.uci.edu/ml/datasets/Carbon+Nanotubes
"""
get_uci_carbon = lambda : pd.read_csv(os.path.join(uci_folder, 'carbon_nanotubes.csv'), delimiter=';', decimal=',')

""" UCI Concrete Compressive Strength Regression Set (1030, 8, 1) 
    https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
"""
get_uci_concrete = lambda : pd.read_excel(os.path.join(uci_folder, 'Concrete_Data.xls'))

""" UCI Energy Efficiency Regression Set (768, 8, 2) 
    https://archive.ics.uci.edu/ml/datasets/Energy+efficiency
"""
get_uci_energy = lambda : pd.read_excel(os.path.join(uci_folder, 'ENB2012_data.xlsx'))

""" UCI Housing Regression Set (506, 13, 1)
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
"""
get_uci_housing = lambda : pd.read_csv(os.path.join(uci_folder, 'housing.data'), delim_whitespace=True, header=None)


""" UCI Kin8m Regression Set (8192, 8, 1)
    https://www.openml.org/search?type=data&sort=runs&id=189
"""
get_uci_kin8m = lambda : pd.DataFrame(arff.loadarff(os.path.join(uci_folder, 'dataset_2175_kin8nm.arff'))[0])

""" UCI Naval Propulsion Plants Regression Set (11934, 16, 2)
    https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
"""
get_uci_naval = lambda : pd.read_csv(os.path.join(uci_folder, 'naval_data.txt'), delim_whitespace=True, header=None)

""" UCI Combined Cycle Power Plant Regression Set (9568, 4, 1)
    https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
"""
get_uci_power = lambda : pd.read_excel(os.path.join(uci_folder, 'Folds5x2_pp.xlsx'))

""" UCI Protein Tertiary Structure Regression Set (45730, 9, 1)
    https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
"""
get_uci_protein = lambda : pd.read_csv(os.path.join(uci_folder, 'CASP.csv'), delimiter=',')


""" UCI Superconductivity Regression Set (21263, 81, 1)
    https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data
"""
get_uci_supercond = lambda : pd.read_csv(os.path.join(uci_folder, 'superconduct.csv'), delimiter=',')

""" UCI Wine Quality Red Regression Set (1599, 11, 1)
    https://archive.ics.uci.edu/ml/datasets/Wine+Quality
"""
get_uci_winer = lambda : pd.read_csv(os.path.join(uci_folder, 'winequality-red.csv'), delimiter=';')

""" UCI Wine Quality White Regression Set (4898, 11, 1)
    https://archive.ics.uci.edu/ml/datasets/Wine+Quality
"""
get_uci_winew = lambda : pd.read_csv(os.path.join(uci_folder, 'winequality-white.csv'), delimiter=';')

""" UCI Yacht Hydrodynamics Regression Set (308, 6, 1)
    https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics
"""
get_uci_yacht = lambda : pd.read_csv(os.path.join(uci_folder, 'yacht_hydrodynamics.data'), delim_whitespace=True, header=None)


###################################################################################################
### UCI DATASET UTILS #############################################################################
uci_sets_dims = {
  'carbon' : (10721, 5, 3),
  'concrete' : (1030, 8, 1),
  'energy' : (768, 8, 2),
  'housing' : (506, 13, 1),
  'kin8m' : (8192, 8, 1),
  'naval' : (11934, 16, 2),
  'power' : (9568, 4, 1),
  'protein' : (45730, 9, 1),
  'supercond' : (21263, 81, 1),
  'winer' : (1599, 11, 1),
  'winew' : (4898, 11, 1),
  'yacht' : (308, 6, 1),
}


def get(set_id : str):
  _, num_input, num_output = uci_sets_dims[set_id]

  data = globals()[f'get_uci_{set_id}']().to_numpy()
  features, labels = data[:, :num_input], data[:, -num_output:]

  return features, labels, uci_sets_dims[set_id]


"""
- PROCESSOR FOR ALL GIVEN SIZE ETC
"""




def get_dataset(get_data):
  # get data and convert to numpy
  data = get_data().to_numpy()

  # normalize (over training data, whiten)
  
  # shape accordingly
  
  # divide in features and labels
  
  pass



if __name__ == '__main__':
  import numpy as np

  features, labels, (num_samples, num_features, num_outputs) = get(set_id='carbon')
  print(features.shape, labels.shape)
  quit()
  
  
  
  data = get_uci_yacht()
  print(data.to_numpy().shape)
  quit()

  data = get_uci_energy().to_numpy()
  normalized_data = (data - data.mean(axis=0)) / data.std(axis=0)

  features, labels = normalized_data[:, :-2], normalized_data[:, -2:, np.newaxis]
  print(features.shape, labels.shape)