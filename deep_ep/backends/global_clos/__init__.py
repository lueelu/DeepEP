"""Global Clos topology entry; the machine selector fixes UDMA transport."""

from ..direct import api

Buffer = api.Buffer
ElasticBuffer = api.ElasticBuffer


def buffer_class(name):
    return api.buffer_class(name)


__all__ = ["Buffer", "ElasticBuffer", "buffer_class"]
