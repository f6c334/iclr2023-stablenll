import tensorflow as tf

from architectures import GaussianSplitMLP, GaussianMLP
# from optimizer_mixins import TFOptimizersModel, TractableOptimizerModel, PitfallsOptimizerModel, NaturalOptimizerModel, AuxiliaryLossTrustRegionsOptimizerModel, TrustRegionsOptimizerModel, TrpTracOptimizerModel, OtherTrpTracOptimizerModel, ThirdTrpTracOptimizerModel
from optimizer_mixins.tfoptimizers import TFOptimizersModel
from optimizer_mixins.pitfalls import PitfallsOptimizerModel
from optimizer_mixins.trustregions import TrueTrustRegionsOptimizerModel, AuxiliaryLossTrustRegionsOptimizerModel, NBackTrustRegionsOptimizerModel, ComparisonNbackTrueTrustRegionsOptimizerModel
from optimizer_mixins.tractable import TractableOptimizerModel
from optimizer_mixins.trptrac import TraptableOptimizerModel, TraptableVariantOptimizerModel, MultipleTraptableOptimizerModel
from optimizer_mixins.conjugate import ConjugateGradientOptimizerModel


###############################################################################
### FINAL COMPLETE MODELS #####################################################
class AdamModel(GaussianSplitMLP, TFOptimizersModel):

  def __init__(self, learning_rate=1e-3, **kwargs):
    super(AdamModel, self).__init__(**kwargs)

    self.learning_rate = learning_rate
    self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


class TractableModel(GaussianSplitMLP, TractableOptimizerModel):

  def __init__(self, learning_rate=1e-3, **kwargs):
    super(TractableModel, self).__init__(**kwargs)

    self.learning_rate = learning_rate
    self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


class TrueTrustRegionModel(GaussianSplitMLP, TrueTrustRegionsOptimizerModel):

  def __init__(self, learning_rate=1e-3, **kwargs):
    super(TrueTrustRegionModel, self).__init__(**kwargs)

    self.learning_rate = learning_rate
    self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


class AuxiliaryLossTrustRegionModel(GaussianSplitMLP, AuxiliaryLossTrustRegionsOptimizerModel):

  def __init__(self, learning_rate=1e-3, **kwargs):
    super(AuxiliaryLossTrustRegionModel, self).__init__(**kwargs)

    self.learning_rate = learning_rate
    self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


class NBackTrustRegionModel(GaussianSplitMLP, NBackTrustRegionsOptimizerModel):

  def __init__(self, learning_rate=1e-3, **kwargs):
    super(NBackTrustRegionModel, self).__init__(**kwargs)

    self.learning_rate = learning_rate
    self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


class ComparisonNBackTrustRegionModel(GaussianSplitMLP, ComparisonNbackTrueTrustRegionsOptimizerModel):

  def __init__(self, learning_rate=1e-3, **kwargs):
    super(ComparisonNBackTrustRegionModel, self).__init__(**kwargs)

    self.learning_rate = learning_rate
    self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


class PitfallsModel(GaussianSplitMLP, PitfallsOptimizerModel):

  def __init__(self, learning_rate=1e-3, **kwargs):
    super(PitfallsModel, self).__init__(**kwargs)

    self.learning_rate = learning_rate
    self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


class TraptableModel(GaussianSplitMLP, TraptableOptimizerModel):

  def __init__(self, learning_rate=1e-3, **kwargs):
    super(TraptableModel, self).__init__(**kwargs)

    self.learning_rate = learning_rate
    self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


class TraptableVariantModel(GaussianSplitMLP, TraptableVariantOptimizerModel):

  def __init__(self, learning_rate=1e-3, **kwargs):
    super(TraptableVariantModel, self).__init__(**kwargs)

    self.learning_rate = learning_rate
    self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

class MultipleTraptableModel(GaussianSplitMLP, MultipleTraptableOptimizerModel):
    def __init__(self, learning_rate=1e-3, **kwargs):
        super(MultipleTraptableModel, self).__init__(**kwargs)

        self.learning_rate = learning_rate
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

class ConjugateGradientModel(GaussianSplitMLP, ConjugateGradientOptimizerModel):

  def __init__(self, learning_rate=1e-3, **kwargs):
    super(ConjugateGradientModel, self).__init__(**kwargs)

    self.learning_rate = learning_rate
    self.compile()
