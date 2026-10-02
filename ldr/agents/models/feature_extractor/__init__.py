from .dummy_extractor import DummyExtractor
from .mlp_extractor import MLPFeatureExtractor
from .latent_mlp_extractor import LatentMLPFeatureExtractor
from .film_extractor import FiLMFeatureExtractor
from .rma_encoder import EnvironmentFactorEncoder

feature_extractor_factory = {}
feature_extractor_factory["DummyExtractor"] = DummyExtractor
feature_extractor_factory["MLPFeatureExtractor"] = MLPFeatureExtractor
feature_extractor_factory["LatentMLPFeatureExtractor"] = LatentMLPFeatureExtractor
feature_extractor_factory["FiLMFeatureExtractor"] = FiLMFeatureExtractor
feature_extractor_factory["EnvironmentFactorEncoder"] = EnvironmentFactorEncoder

__all__ = ["feature_extractor_factory"]
