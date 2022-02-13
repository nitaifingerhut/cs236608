from .autorec import Autorec
from .libfm import LibFM_MLHB
from .knn import KNNRecommender
from .temporal_autorec import TemporalAutorec

RECOMMENDERS = dict(autorec=Autorec, libfm=LibFM_MLHB, knn=KNNRecommender, temporal_autorec=TemporalAutorec)
