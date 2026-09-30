<!--
Copyright (c) 2026, Lu Lu
Modified by zhu-mingzhe71 2026
-->

EpGmmFused follows the root repository's licensing policy. Material owned by
Lu Lu is covered by the root BSD 2-Clause license; migrated source retains its
original notices and license.

`upstream.json` records a temporary development snapshot and source hashes before
migration. It is historical provenance, not a released dependency, a current file
manifest, or a separate license selection for this component.

Changes in this repository: root build integration; package and dispatcher namespace;
public operator aliases; shared C++ SHMEM bootstrap; ProcessGroup UID exchange and
lifecycle checks; independent generalized tests, performance and profiler runners.
Device computation and UDMA completion protocols are retained from the PR.

Third-party code retains its copyright and license notices, including the CatCCOS
overlay (csrc/catccos_overlay/LICENSE in the source distribution). The wheel
includes the licenses from the pinned CatCCOS, CATLASS and
SHMEM checkouts under licenses/catccos, licenses/catlass and licenses/shmem.
The adopted DeepSeek API license is packaged by the root project.
