"""Single-host MESH_PCIE policy over the existing MTE V1/V2 kernels."""

from pathlib import Path
import uuid

from ..direct import api
from ..._machine import gather_records


def host_identity():
    # Hostname/IP can be shared or differ across containers. A Linux boot ID
    # identifies the current OS instance; unavailable evidence fails closed.
    boot = uuid.UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    if boot.int == 0:
        raise ValueError("Linux boot ID is unset")
    return str(boot)


def prepare(group, options):
    if group is None:
        raise ValueError("mesh_pcie requires an EP ProcessGroup or MPI communicator")
    error, host = None, None
    try:
        if options.get("udma_schedule", "peer") != "peer":
            raise ValueError("mesh_pcie does not support the UDMA token schedule")
        host = host_identity()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    records = gather_records(group, {"host": host, "error": error})
    errors = [(rank, record["error"]) for rank, record in enumerate(records) if record["error"]]
    if errors:
        raise RuntimeError(f"mesh_pcie preflight failed collectively: {errors}")
    if len({record["host"] for record in records}) != 1:
        raise ValueError("mesh_pcie MTE requires all EP ranks on the same host")


class _MappedMte:
    def _validate_native_mappings(self, group):
        errors = []
        try:
            errors = list(self._native.mte_mapping_errors())
        except Exception as exc:
            errors = [f"mapping query failed (rebuild the native extension): {type(exc).__name__}: {exc}"]
        try:
            records = gather_records(group, errors)
            failures = [(rank, errors) for rank, errors in enumerate(records) if errors]
            if failures:
                raise RuntimeError(f"mesh_pcie MTE mapping check failed collectively: {failures}")
        except Exception as error:
            # No DeepEP operation has launched. Avoid a device/SHMEM barrier on
            # precisely the peer mappings which were rejected or are unknown.
            try:
                self._native.abort_initialization()
            except Exception as cleanup_error:
                error.add_note(f"native initialization cleanup: {cleanup_error}")
            finally:
                self._destroyed = True
                self._native = None
            raise


class Buffer(_MappedMte, api.Buffer):
    pass


class ElasticBuffer(_MappedMte, api.ElasticBuffer):
    pass


def buffer_class(name):
    return {"Buffer": Buffer, "ElasticBuffer": ElasticBuffer}[name]
