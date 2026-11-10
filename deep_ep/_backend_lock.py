"""Process-local SHMEM construction lock shared by Mesh+Clos and Clos adapters."""

from threading import RLock

LOCK = RLock()
