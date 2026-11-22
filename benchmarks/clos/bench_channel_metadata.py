"""EP8/16, eight AIV: channel-metadata off/on/on/off with one Buffer per rank.

Shares the stage benchmark's route seed, API/profile separation, precision
checks, library/huge-page audit and raw sample exports. Cached is a negative
control: the native channel-metadata gate must remain off on cached calls.
"""

import sys

from bench_dispatch_stages import main

if __name__ == "__main__":
    main(
        [
            "--cores",
            "8",
            "--cached-channel-modes",
            "0",
            "--metadata-modes",
            "0",
            "1",
            "--trace-samples",
            "0",
            "--flush-l2",
            *sys.argv[1:],
        ]
    )
