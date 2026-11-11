"""Resolve memory options for an explicitly selected Elastic backend."""


def elastic_backend_options(requested, *, combine_only, huge_pages):
    """Preserve public MTE defaults; LL launchers explicitly select RDMA."""
    if requested not in (None, "mte", "rdma"):
        raise ValueError("DEEPEP_LL_BACKEND must be mte or rdma")
    backend = requested or "mte"
    if huge_pages is None:
        huge_pages = backend != "rdma"
    if type(huge_pages) is not bool:
        raise ValueError("use_1g_huge_pages must be bool or None")
    if backend == "rdma" and (huge_pages or combine_only):
        raise ValueError("RDMA LL requires use_1g_huge_pages=False and combine_only=False")
    return backend, huge_pages
