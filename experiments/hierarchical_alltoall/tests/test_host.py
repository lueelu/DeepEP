#!/usr/bin/env python3
"""CPU protocol and launcher checks; does not compile AscendC or exercise any NPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HostTests(unittest.TestCase):
    def test_protocol(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler, 'A C++17 compiler is needed for the protocol model')
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / 'protocol'
            subprocess.run([compiler, '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-I' + str(ROOT), '-I' + str(ROOT.parents[1] / 'csrc'),
                            str(ROOT / 'tests/protocol_test.cpp'), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True, timeout=120)

    def test_public_swizzle(self):
        catlass = Path(os.environ.get('CATLASS_SOURCE_PATH', ROOT.parents[1] / 'third-party/catlass'))
        if not (catlass / 'include/catlass/catlass.hpp').is_file():
            self.skipTest('Set CATLASS_SOURCE_PATH to enable real Catlass/scheduler arithmetic checks')
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / 'swizzle'
            subprocess.run([compiler, '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-I' + str(ROOT.parents[1] / 'csrc'),
                            '-isystem', str(catlass / 'include'), str(ROOT / 'tests/swizzle_test.cpp'),
                            '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True, timeout=30)

    def test_block_completion(self):
        catlass = Path(os.environ.get('CATLASS_SOURCE_PATH', ROOT.parents[1] / 'third-party/catlass'))
        if not (catlass / 'include/catlass/catlass.hpp').is_file():
            self.skipTest('Set CATLASS_SOURCE_PATH for block completion tests')
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / 'block'
            subprocess.run([compiler, '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-I' + str(ROOT / 'tests/block_stubs'),
                            '-I' + str(ROOT.parents[1] / 'csrc'), '-isystem', str(catlass / 'include'),
                            str(ROOT / 'tests/block_completion_test.cpp'), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True, timeout=30)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.build = base / 'build'
        self.build.mkdir()
        sdk = base / 'sdk'
        for name in ('include/shmem.h', 'include/host/shmem_host_def.h',
                     'include/host/init/shmem_host_init.h', 'lib/libshmem.so'):
            path = sdk / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fake SDK for shell test\n')
        (sdk / 'src/device').mkdir(parents=True)
        (self.build / 'hierarchical_alltoall').write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "$*"\n'
            'if [ "${FAIL_RANK:-none}" = "$2" ]; then exit 7; fi\n')
        (self.build / 'hierarchical_alltoall').chmod(0o755)
        self.env = dict(os.environ, SHMEM_ROOT=str(sdk), ASCEND_HOME_PATH=str(base / 'cann'),
                        CATLASS_SOURCE_PATH=str(ROOT), GROUP_SIZE='16', COMM_BLOCK_M='64',
                        SHMEM_RDMA_BACKEND='HNS_1825', SHMEM_HNS_1825_PI_ON_CHIP='1',
                        BUILD_DIR=str(self.build), NNODES='2', NODE_RANK='1',
                        RANK0_IP='90.90.93.84', QP_NUM='4', RUN_ID='test',
                        DEVICES=','.join(map(str, range(16))), PYTHON=shutil.which('python3'),
                        M='257', N='1024', WINDOW_M='64', VERIFY_READY='1', PORT='8666')
        subprocess.run([self.env['PYTHON'], str(ROOT / 'scripts/config.py'), 'record', str(self.build)],
                       env=self.env, check=True)

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, **updates):
        return subprocess.run(['bash', '+o', 'posix', str(ROOT / 'scripts/run.sh')],
                              env=dict(self.env, **updates), capture_output=True, text=True, timeout=15)

    def test_dual_node_arguments(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        logs = self.build / 'logs/test-node1'
        self.assertEqual(len(list(logs.glob('rank*.log'))), 16)
        self.assertEqual((logs / 'rank16.log').read_text().strip(),
                         '32 16 tcp://90.90.93.84:8666 257 1024 64 0 4 1')
        self.assertEqual((logs / 'rank31.log').read_text().strip(),
                         '32 31 tcp://90.90.93.84:8666 257 1024 64 15 4 1')
        self.assertNotEqual(self.launch().returncode, 0) # Cannot overwrite results.

    def test_failure_propagation(self):
        self.assertNotEqual(self.launch(FAIL_RANK='20').returncode, 0)

    def test_invalid_options(self):
        for options in ({'RANK0_IP': ''}, {'RANK0_IP': '90.90.93.84^Cash'}, {'QP_NUM': '3'},
                        {'N': '33'}, {'WINDOW_M': '0'}, {'DEVICES': '0,1'},
                        {'DEVICES': ','.join(['0'] * 16)}, {'NODE_RANK': '2'}):
            with self.subTest(options=options):
                self.assertNotEqual(self.launch(**options).returncode, 0)
        self.assertFalse((self.build / 'logs').exists())

    def test_single_node(self):
        result = self.launch(NNODES='1', NODE_RANK='0', DEVICES='0,1', QP_NUM='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.build / 'logs/test-node0/rank1.log').exists())

    def test_build_dependency_selection(self):
        # Check CLI/environment forwarding only; this is not a CANN/CMake build.
        bin_dir = Path(self.temp.name) / 'bin'
        bin_dir.mkdir()
        capture = Path(self.temp.name) / 'cmake-args'
        cmake = bin_dir / 'cmake'
        cmake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$CAPTURE"\n')
        cmake.chmod(0o755)
        for mode in ('explicit', 'legacy', 'default'):
            env = dict(self.env, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                       CXX='/usr/bin/true', CAPTURE=str(capture))
            env.pop('CATCCOS_SOURCE_PATH', None)
            if mode == 'explicit':
                expected = env['CATLASS_SOURCE_PATH']
            else:
                env.pop('CATLASS_SOURCE_PATH', None)
                if mode == 'legacy':
                    env['CATCCOS_SOURCE_PATH'] = str(Path(self.temp.name) / 'catccos')
                    expected = env['CATCCOS_SOURCE_PATH'] + '/3rdparty/catlass'
                else:
                    expected = str(ROOT.parents[1] / 'third-party/catlass')
            capture.write_text('')
            result = subprocess.run(['bash', str(ROOT / 'scripts/build.sh')], env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('-DCATLASS_SOURCE_PATH=' + expected, capture.read_text().splitlines())
            self.assertNotIn('-DCATCCOS_SOURCE_PATH=', capture.read_text())
            config = json.loads((self.build / 'build_config.json').read_text())
            self.assertEqual(config['catlass'], str(Path(expected).resolve()))

    def test_bisheng_selection(self):
        # Exercise the real build script with fake compiler/CMake executables.
        base = Path(self.temp.name)
        bin_dir = base / 'bin'
        bin_dir.mkdir()
        capture = base / 'compiler-args'
        cmake = bin_dir / 'cmake'
        cmake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$CAPTURE"\n')
        cmake.chmod(0o755)
        env = dict(self.env, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'], CAPTURE=str(capture))
        env.pop('CXX', None)
        cann = Path(env['ASCEND_HOME_PATH'])
        candidates = [cann / 'tools/bisheng_compiler/bin/bisheng',
                      cann / 'compiler/ccec_compiler/bin/bisheng',
                      cann / (os.uname().machine + '-linux/bin/bisheng')]
        for compiler in candidates:
            compiler.parent.mkdir(parents=True, exist_ok=True)
            compiler.write_text('#!/bin/sh\nexit 0\n')
            compiler.chmod(0o755)

        def check(expected, explicit=None):
            capture.write_text('')
            options = dict(env)
            if explicit is not None:
                options['CXX'] = explicit
            result = subprocess.run(['bash', str(ROOT / 'scripts/build.sh')], env=options,
                                    capture_output=True, text=True, timeout=15)
            if expected is None:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Missing Bisheng', result.stderr)
                self.assertEqual(capture.read_text(), '')
            else:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('-DCMAKE_CXX_COMPILER=' + str(expected), capture.read_text().splitlines())

        check(candidates[1], str(candidates[1])) # Explicit CXX takes precedence.
        check(None, str(base / 'missing')) # Do not silently ignore a broken override.
        for compiler in candidates:
            check(compiler)
            compiler.unlink()
        check(None)
        named = bin_dir / 'custom-bisheng'
        named.write_text('#!/bin/sh\nexit 0\n')
        named.chmod(0o755)
        check(named, 'custom-bisheng')

    def test_sdk_changed(self):
        Path(self.env['SHMEM_ROOT'], 'lib/libshmem.so').write_text('replaced SDK')
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SDK changed after build', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
