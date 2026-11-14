"""Pinned SHMEM release checks; fake libraries/ACL/Torch never qualify NPU execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third-party/shmem"
LOCK = json.loads((ROOT / "dependencies.lock.json").read_text())["shmem"]
spec = importlib.util.spec_from_file_location("record_shmem_sdk", ROOT / "scripts/record_shmem_sdk.py")
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


@unittest.skipUnless((SOURCE / "include/shmem.h").is_file(), "Initialize the SHMEM submodule")
class ReleaseTests(unittest.TestCase):
    def test_locked_release_headers(self):
        self.assertEqual(LOCK["ref"], "refs/tags/v1.8.0")
        revision = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
        self.assertEqual(revision, LOCK["revision"])
        for name, digest in LOCK["header_sha256_lf"].items():
            data = (SOURCE / "include" / name).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, name)

    def test_sdk_identity_rejects_old_revision_and_changed_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = Path(temporary)
            for name in LOCK["header_sha256_lf"]:
                target = sdk / "include" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SOURCE / "include" / name, target)
            (sdk / "lib").mkdir()
            (sdk / "lib/libshmem.so").write_bytes(b"test fixture, not a usable SHMEM library")
            with self.assertRaisesRegex(ValueError, "revision"):
                recorder.record(sdk, "836d3fd6f9c2f3b10a6acac36ccfd531b104d4a7")
            recorder.record(sdk, LOCK["revision"])
            recorder.check(sdk)
            header = sdk / "include/host/shmem_host_def.h"
            header.write_bytes(header.read_bytes() + b"\n// changed\n")
            with self.assertRaisesRegex(ValueError, "header mismatch"):
                recorder.check(sdk)

    def test_raw_remote_wqe_and_blocking_copy(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("A C++17 compiler is required")
        common = (SOURCE / "src/device/shmemi_device_common.hpp").read_text()
        udma = (SOURCE / "src/device/gm2gm/engine/shmem_device_udma.hpp").read_text()
        bodies = []
        for text, symbol, templated in (
            (common, "aclshmemi_debug_assert_remote_mr_addr", False),
            (common, "aclshmemi_remote_mr_addr", False),
            (udma, "aclshmemi_udma_fill_sqe_base_ctx", True),
        ):
            start = text.index(symbol + "(")
            start = text.rfind("template <" if templated else "ACLSHMEM_DEVICE", 0, start)
            opening = text.index("{", start)
            depth, end = 1, opening + 1
            while depth:
                depth += (text[end] == "{") - (text[end] == "}")
                end += 1
            bodies.append(text[start:end])
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "sdk_sqe.inc").write_text("\n".join(bodies))
            (work / "shmem.h").write_text("// SDK bodies/types supplied by the CPU test.\n")
            binary = work / "wqe"
            subprocess.run([
                compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                "-I" + str(work), "-I" + str(ROOT / "csrc"),
                str(ROOT / "tests/runtime/shmem_wqe_test.cpp"), "-o", str(binary)
            ], check=True, timeout=60)
            subprocess.run([str(binary)], check=True, timeout=10)

    def test_production_host_sources_against_release_headers(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("A C++17 compiler is required")
        with tempfile.TemporaryDirectory() as temporary:
            stub = Path(temporary)
            (stub / "acl").mkdir()
            # Only ACL, platform and Torch are stubbed; SHMEM headers are real.
            (stub / "acl/acl.h").write_text((ROOT / "tests/runtime/stubs/acl/acl.h").read_text() + """
using aclrtStream = void*;
using aclError = int;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 2;
int aclInit(const char*);
int aclFinalize();
int aclrtSetDevice(int);
int aclrtResetDevice(int);
int aclrtSetCurrentContext(aclrtContext);
int aclrtCreateStream(aclrtStream*);
int aclrtDestroyStream(aclrtStream);
int aclrtSynchronizeStream(aclrtStream);
int aclrtSynchronizeDevice();
int aclrtMalloc(void**, size_t, int);
int aclrtFree(void*);
int aclrtMemcpy(void*, size_t, const void*, size_t, int);
const char* aclGetRecentErrMsg();
""")
            (stub / "acl/acl_rt.h").write_text('#pragma once\n#include "acl/acl.h"\n')
            (stub / "torch").mkdir()
            (stub / "torch/custom_class.h").write_text(
                '#pragma once\nnamespace torch::jit { class CustomClassHolder {}; }\n')
            (stub / "tiling/platform").mkdir(parents=True)
            (stub / "tiling/platform/platform_ascendc.h").write_text("""
namespace platform_ascendc {
struct PlatformAscendCManager {
    static PlatformAscendCManager* GetInstance();
    unsigned GetCoreNumAic();
    unsigned GetCoreNumAiv();
};
}
""")
            for name in ("csrc/runtime/session.cpp", "csrc/direct/runtime/shmem_instance.cpp",
                         "experiments/hierarchical_alltoall/main.cpp"):
                with self.subTest(source=name):
                    subprocess.run([
                        compiler, "-std=c++17", "-fsyntax-only", "-Wall", "-Wextra", "-Werror",
                        "-Wno-unused-parameter", "-I" + str(stub), "-I" + str(SOURCE / "include"),
                        "-I" + str(SOURCE / "src"), "-I" + str(ROOT / "csrc"),
                        "-I" + str(ROOT / "csrc/direct"), "-DHIERARCHICAL_GROUP_SIZE=16",
                        "-DHIERARCHICAL_SHMEM_RUNTIME_CONFIG=1", str(ROOT / name)
                    ], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
