#!/usr/bin/env python3
"""Record the selected SDK/build inputs and reject accidental runtime SDK switches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def git_head(path):
    result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['record', 'check'])
    parser.add_argument('build', type=Path)
    args = parser.parse_args()
    path = args.build / 'build_config.json'
    if args.action == 'record':
        sdk = Path(os.environ['SHMEM_ROOT']).resolve()
        files = ['include/shmem.h', 'include/host/shmem_host_def.h',
                 'include/host/init/shmem_host_init.h', 'lib/libshmem.so']
        if not (sdk / 'src/device').is_dir():
            raise ValueError('SHMEM_ROOT must contain installed include/src/lib, not the source or install parent')
        catlass = Path(os.environ['CATLASS_SOURCE_PATH']).resolve()
        config = dict(shmem_root=str(sdk), cann=os.environ['ASCEND_HOME_PATH'],
                      catlass=str(catlass), catlass_commit=git_head(catlass),
                      group_size=int(os.environ['GROUP_SIZE']),
                      comm_block_m=int(os.environ['COMM_BLOCK_M']),
                      rdma_backend=os.environ['SHMEM_RDMA_BACKEND'],
                      pi_on_chip=os.environ.get('SHMEM_HNS_1825_PI_ON_CHIP', ''),
                      sdk_hashes={name: digest(sdk / name) for name in files},
                      executable_sha256=digest(args.build / 'hierarchical_alltoall'))
        path.write_text(json.dumps(config, indent=2) + '\n')
    else:
        config = json.loads(path.read_text())
        sdk = Path(config['shmem_root'])
        if 'SHMEM_ROOT' in os.environ and Path(os.environ['SHMEM_ROOT']).resolve() != sdk:
            raise ValueError('SHMEM_ROOT differs from the SDK used to build; rebuild or select the recorded SDK')
        for name, expected in config['sdk_hashes'].items():
            if digest(sdk / name) != expected:
                raise ValueError('SDK changed after build: ' + name + '; rebuild before running')
        if digest(args.build / 'hierarchical_alltoall') != config['executable_sha256']:
            raise ValueError('Executable changed after build; run build.sh again')
        # One field per line: run.sh reads without evaluating shell code.
        for key in ('group_size', 'shmem_root', 'cann', 'rdma_backend', 'pi_on_chip'):
            print(config[key])


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit('[ERROR] ' + str(error))
