# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Public buffer declarations; no native resources are initialized on import."""

from .elastic import ElasticBuffer, EPHandle
from .legacy import Buffer

__all__ = ["Buffer", "ElasticBuffer", "EPHandle"]
