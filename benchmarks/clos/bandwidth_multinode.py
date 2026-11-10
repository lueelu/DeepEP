"""CPU-only coordination and bounded raw-file transfer for bandwidth benchmarks.

No SSH, shared filesystem, torch/NPU imports, or remote command execution.
The socket carries JSON and file bytes only. Use a trusted benchmark network.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import socket
import struct
import time

from stage_metrics import file_identity, write_json


CHUNK_BYTES = 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or "\\" in relative
        or ":" in relative
        or any(p in ("", ".", "..") for p in relative.split("/"))
    ):
        raise ValueError(f"unsafe relative artifact path: {relative!r}")
    target = Path(root).joinpath(*path.parts)
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("artifact escapes collection root")
    return target


def content_identity(identity, repo):
    """Source checkout identity without machine-specific absolute paths."""
    root = Path(repo).resolve()
    return dict(
        revision=identity["revision"],
        golden={k: identity["golden"][k] for k in ("size", "sha256")},
        sources=sorted(
            (str(Path(f["path"]).resolve().relative_to(root)).replace("\\", "/"), f["size"], f["sha256"])
            for f in identity["sources"]
        ),
    )


def library_identity(identity):
    """Compare the actually loaded bytes, even when installation paths differ."""
    entries = [(Path(f["path"]).name, f["size"], f["sha256"]) for f in identity["libraries"]]
    if len({e[0] for e in entries}) != len(entries):
        raise ValueError("ambiguous loaded library basenames")
    return dict(
        torch=identity["torch"],
        torch_npu=identity["torch_npu"],
        python_module={k: identity["python_module"][k] for k in ("size", "sha256")},
        libraries=sorted(entries),
    )


def validate_peers(local, remote):
    if {local["node_rank"], remote["node_rank"]} != {0, 1}:
        raise ValueError("expected exactly node ranks 0 and 1")
    # JSON roundtrip normalizes tuples to lists; paths and first_device may differ.
    for key in ("contract", "source", "topk_idx_bytes"):
        if json.dumps(local[key], sort_keys=True) != json.dumps(remote[key], sort_keys=True):
            raise ValueError(f"two-node {key} mismatch; no NPU job started")


class Peer:
    def __init__(self, connection, node_rank, timeout=30):
        self.connection = connection
        self.node_rank = node_rank
        self.connection.settimeout(timeout)

    @classmethod
    def connect(cls, address, port, node_rank, rendezvous_timeout):
        deadline = time.monotonic() + rendezvous_timeout
        if node_rank == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("0.0.0.0", port))
                listener.listen(1)
                listener.settimeout(rendezvous_timeout)
                connection, _ = listener.accept()
        else:
            # Rendezvous connection only: never retries a launched benchmark job.
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("node 0 coordination endpoint did not become available")
                try:
                    connection = socket.create_connection((address, port), timeout=min(2, remaining))
                    break
                except (ConnectionRefusedError, TimeoutError):
                    time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        return cls(connection, node_rank)

    def close(self):
        self.connection.close()

    def _read(self, count):
        pieces = bytearray()
        while len(pieces) < count:
            part = self.connection.recv(min(CHUNK_BYTES, count - len(pieces)))
            if not part:
                raise ConnectionError("peer disconnected; stop this suite")
            pieces.extend(part)
        return bytes(pieces)

    def send_json(self, data):
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_JSON_BYTES:
            raise ValueError("coordination JSON exceeds limit")
        self.connection.sendall(struct.pack("!I", len(encoded)) + encoded)

    def receive_json(self):
        (size,) = struct.unpack("!I", self._read(4))
        if not 0 < size <= MAX_JSON_BYTES:
            raise ValueError("invalid coordination frame size")
        return json.loads(self._read(size))

    def exchange(self, stage, value):
        # Fixed ordering avoids two simultaneous large sends blocking one another.
        outgoing = dict(stage=stage, value=value)
        if self.node_rank == 0:
            self.send_json(outgoing)
            incoming = self.receive_json()
        else:
            incoming = self.receive_json()
            self.send_json(outgoing)
        if incoming.get("stage") != stage:
            raise ValueError(f"peer stage mismatch at {stage}")
        return incoming["value"]

    def require_both(self, stage, error=None):
        other = self.exchange(stage, dict(error=error))
        if error or other["error"]:
            raise RuntimeError(f"{stage}: local={error}; peer={other['error']}")

    def transfer(self, source, destination, manifest, timeout):
        """Node 1 streams reviewed files to node 0; no tar extraction or pickle."""
        deadline = time.monotonic() + timeout
        old_timeout = self.connection.gettimeout()

        def remaining():
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError("raw-artifact collection deadline exceeded")
            self.connection.settimeout(min(30, value))

        try:
            remaining()
            if self.node_rank == 1:
                self.send_json(manifest)
            else:
                manifest = self.receive_json()
            if not isinstance(manifest, list) or not manifest:
                raise ValueError("empty/invalid artifact manifest")
            seen = set()
            for item in manifest:
                relative, size = item["relative"], item["size"]
                if relative in seen or type(size) is not int or size < 0:
                    raise ValueError("duplicate artifact or invalid byte count")
                seen.add(relative)
                path = safe_path(source if self.node_rank == 1 else destination, relative)
                digest = hashlib.sha256()
                remaining()
                if self.node_rank == 1:
                    with path.open("rb") as stream:
                        pending = size
                        while pending:
                            remaining()
                            block = stream.read(min(CHUNK_BYTES, pending))
                            if not block:
                                raise ValueError("source artifact became shorter")
                            digest.update(block)
                            self.connection.sendall(block)
                            pending -= len(block)
                        if stream.read(1):
                            raise ValueError("source artifact grew")
                    if digest.hexdigest() != item["sha256"]:
                        raise ValueError("source artifact changed during transfer")
                    remaining()
                    if self.receive_json() != dict(received=relative, sha256=item["sha256"]):
                        raise ValueError("peer did not verify artifact")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + ".download")
                    if path.exists():
                        raise FileExistsError(f"refusing to overwrite collected artifact: {path}")
                    with temporary.open("xb") as stream:
                        pending = size
                        while pending:
                            remaining()
                            block = self._read(min(CHUNK_BYTES, pending))
                            digest.update(block)
                            stream.write(block)
                            pending -= len(block)
                    if digest.hexdigest() != item["sha256"]:
                        raise ValueError(f"artifact hash mismatch: {relative}")
                    temporary.rename(path)
                    remaining()
                    self.send_json(dict(received=relative, sha256=item["sha256"]))
            if self.node_rank == 0:
                write_json(Path(destination) / "collection.json", dict(passed=True, files=manifest))
            return manifest
        finally:
            self.connection.settimeout(old_timeout)


def collection_manifest(case_path, ranks):
    """Completed ranks only; include every raw CSV/trace/info cited in results."""
    root = Path(case_path).resolve()
    paths = {root / name for name in ("command.json", "resources.before.txt", "launcher_exit.json", "run.log")}
    status = json.loads((root / "launcher_exit.json").read_text(encoding="utf-8"))
    if status["code"] != 0 or status.get("suite_aborted"):
        raise ValueError("cannot register a failed launcher as complete")
    for rank in ranks:
        rank_dir = root / f"rank{rank}"
        ex = json.loads((rank_dir / "exit.json").read_text(encoding="utf-8"))
        record = json.loads((rank_dir / "result.json").read_text(encoding="utf-8"))
        if ex["code"] != 0 or record["rank"] != rank or not record["destroyed"] or not record["precision_passed"]:
            raise ValueError(f"incomplete rank {rank}")
        paths.update(
            rank_dir / name
            for name in ("process.json", "precision.json", "result.json", "input_routes.json", "exit.json")
        )
        for op, data in record["operators"].items():
            paths.add(rank_dir / op / "measurement.json")
            for artifact in data["profile"]["artifacts"]:
                path = Path(artifact["path"]).resolve()
                if not path.is_relative_to(rank_dir.resolve()):
                    raise ValueError("profiler artifact is outside its rank directory")
                current = file_identity(path)
                if any(current[k] != artifact[k] for k in ("size", "sha256")):
                    raise ValueError("completed profiler artifact changed")
                paths.add(path)
    result = []
    for path in sorted(paths):
        identity = file_identity(path)
        result.append(
            dict(relative=path.relative_to(root).as_posix(), size=identity["size"], sha256=identity["sha256"])
        )
    return result


class PeerGroup:
    """Node-0 coordinator for 2..8 nodes, using the existing bounded wire format.

    Only rendezvous connections retry. A disconnected or failing job aborts the
    whole group; no task is resubmitted. Raw files transfer one node at a time.
    """

    def __init__(self, peers, node_rank, nnodes):
        self.peers, self.node_rank, self.nnodes = peers, node_rank, nnodes
        self.connection = self  # run_case/collection use settimeout on this facade.

    @classmethod
    def connect(cls, address, port, node_rank, timeout, nnodes):
        peers = {}
        try:
            if node_rank == 0:
                deadline = time.monotonic() + timeout
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    listener.bind(("0.0.0.0", port))
                    listener.listen(nnodes - 1)
                    while len(peers) < nnodes - 1:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("not all benchmark nodes joined")
                        listener.settimeout(remaining)
                        connection, _ = listener.accept()
                        peer = Peer(connection, 0, remaining)
                        try:
                            hello = peer.receive_json()
                            rank = hello.get("node_rank")
                            if (
                                type(rank) is not int
                                or not 1 <= rank < nnodes
                                or rank in peers
                                or hello.get("nnodes") != nnodes
                            ):
                                raise ValueError("invalid/duplicate benchmark node")
                            peers[rank] = peer
                        except BaseException:
                            peer.close()
                            raise
            else:
                peer = Peer.connect(address, port, 1, timeout)
                peers[0] = peer
                peer.send_json(dict(node_rank=node_rank, nnodes=nnodes))
            group = cls(peers, node_rank, nnodes)
            group.settimeout(timeout)
            group.exchange_all("connected", node_rank)
            group.settimeout(30)
            return group
        except BaseException:
            for peer in peers.values():
                peer.close()
            raise

    def settimeout(self, timeout):
        for peer in self.peers.values():
            peer.connection.settimeout(timeout)

    def close(self):
        for peer in self.peers.values():
            peer.close()

    def exchange_all(self, stage, value):
        if self.node_rank == 0:
            values = [value] + [None] * (self.nnodes - 1)
            for rank in range(1, self.nnodes):
                incoming = self.peers[rank].receive_json()
                if incoming.get("stage") != stage:
                    raise ValueError(f"node {rank}: stage mismatch at {stage}")
                values[rank] = incoming["value"]
            for peer in self.peers.values():
                peer.send_json(dict(stage=stage, values=values))
            return values
        peer = self.peers[0]
        peer.send_json(dict(stage=stage, value=value))
        incoming = peer.receive_json()
        if incoming.get("stage") != stage or len(incoming.get("values", [])) != self.nnodes:
            raise ValueError(f"coordinator stage mismatch at {stage}")
        return incoming["values"]

    def exchange(self, stage, value):
        """Aggregate run_case's exit/timeout snapshot across every node."""
        values = self.exchange_all(stage, value)
        failed = next((v["code"] for v in values if v["code"] not in (None, 0)), None)
        return dict(
            code=failed if failed is not None else (0 if all(v["code"] == 0 for v in values) else None),
            timed_out=any(v["timed_out"] for v in values),
        )

    def require_both(self, stage, error=None):
        errors = self.exchange_all(stage, error)
        if any(errors):
            raise RuntimeError(f"{stage}: node errors={errors}")

    def collect(self, path, manifest, timeout):
        # Idle nodes wait while another node transfers. Each transfer retains its
        # own finite deadline and independently verifies every file's size/hash.
        self.settimeout(timeout * self.nnodes)
        for rank in range(1, self.nnodes):
            self.require_both(f"collect/node{rank}/ready")
            if self.node_rank == 0:
                self.peers[rank].transfer(path, path / f"peer_node{rank}", None, timeout)
            elif self.node_rank == rank:
                self.peers[0].transfer(path, path, manifest, timeout)
            self.require_both(f"collected/node{rank}")
        self.settimeout(30)
