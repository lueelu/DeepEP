"""V2 automatic full-CLOS policy benchmark (one EP per invocation).

Defaults: 32 AIVs, BF16, uniform routes, no expansion, T8192/H6144/K8/E32.
Checks precision, then profiles dispatch, cached dispatch and both combines.
L2 eviction is enabled; weighted combine has no bias by default.
No scheduling knobs are supplied to ElasticBuffer: udma_auto_policy=True
selects the actual parameters, which are printed and saved in report.json.

Single node (EP4/8/16):
  python -u benchmarks/clos/bench_v2_auto_bandwidth.py --ep-size 8 --output auto_ep8

EP32 defaults to two nodes with 16 NPUs each. Run on BOTH nodes, with node-rank
0 on the master and 1 on the other node; use identical run-id/shape/options:
  python -u benchmarks/clos/bench_v2_auto_bandwidth.py --ep-size 32 --node-rank 0 \
    --master-addr <NODE0_IP> --master-port 29600 --run-id auto_ep32_01 --output auto_ep32_01

Use --aiv to choose another even core count; insufficient cores raise an error.
Node0 prints the complete summary. notify/reduce are separate from comm bandwidth.
--report-only <output-directory> regenerates the report without NPU execution.
"""

import bench_v2_bandwidth as v2


def arguments(argv=None):
    return v2.arguments(argv, description=__doc__, auto_policy=True)


if __name__ == "__main__":
    options = arguments()
    if options.report_only:
        v2.report(options.report_only)
    elif options.worker:
        v2.worker(options)
    else:
        v2.launch(options, entrypoint=__file__)
