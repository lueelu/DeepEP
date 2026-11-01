import unittest
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

from deep_ep.backends.clos.buffer import _prepare_shmem_bootstrap


class BootstrapTest(unittest.TestCase):
    def prepare(self, native, gather=None, rank=0, endpoint=None, comm=None):
        with (
            patch("deep_ep.backends.clos.buffer.load_native_module", return_value=native),
            patch(
                "deep_ep.backends.clos.buffer._bootstrap_all_gather",
                side_effect=gather or (lambda group, comm, size, value: [value] * size),
            ),
        ):
            return _prepare_shmem_bootstrap(object(), comm, rank, 2, 64, "udma", False, endpoint)

    def test_root_generates_and_active_session_reuses_uid(self):
        native = Mock()
        native.get_shmem_bootstrap.return_value = (0, [])
        native.get_shmem_unique_id.return_value = [1, 2, 3]
        self.assertEqual(self.prepare(native)[0], [1, 2, 3])
        native.get_shmem_bootstrap.return_value = (1, [1, 2, 3])
        self.assertEqual(self.prepare(native)[0], [1, 2, 3])
        native.get_shmem_unique_id.assert_called_once()

    def test_nonroot_uses_root_uid(self):
        native = Mock()
        native.get_shmem_bootstrap.return_value = (0, [])

        def gather(group, comm, size, value):
            return [([4, 5], None), None] if value is None else [value] * size

        self.assertEqual(self.prepare(native, gather, rank=1)[0], [4, 5])
        native.get_shmem_unique_id.assert_not_called()

    def test_uid_failure_does_not_fall_back_to_inherited_endpoint(self):
        native = Mock()
        native.get_shmem_bootstrap.return_value = (0, [])
        native.get_shmem_unique_id.side_effect = RuntimeError("UID creation failed")
        with (
            patch.dict(os.environ, {"ASCEND_DEEPEP_SHMEM_IP_PORT": "tcp://127.0.0.1:30000"}),
            self.assertRaisesRegex(RuntimeError, "EP rank 0.*UID creation failed"),
        ):
            self.prepare(native)
        native.get_shmem_unique_id.assert_called_once()

    def test_mismatched_bootstrap_modes_are_rejected_before_uid_creation(self):
        native = Mock()
        native.get_shmem_bootstrap.return_value = (0, [])

        def gather(group, comm, size, value):
            options = list(value[0])
            options[3] = "tcp://127.0.0.1:30000"
            return [value, (tuple(options), 0, [])]

        with self.assertRaisesRegex(ValueError, "all EP ranks"):
            self.prepare(native, gather)
        native.get_shmem_unique_id.assert_not_called()

    def test_explicit_endpoint_does_not_generate_uid(self):
        native = Mock()
        native.get_shmem_bootstrap.return_value = (0, [])
        self.assertEqual(self.prepare(native, endpoint="tcp://127.0.0.1:30000")[0], [])
        native.get_shmem_unique_id.assert_not_called()

    def test_root_error_is_shared_and_different_active_group_rejected(self):
        native = Mock()
        native.get_shmem_bootstrap.return_value = (0, [])
        native.get_shmem_unique_id.side_effect = RuntimeError("no endpoint")
        with self.assertRaisesRegex(RuntimeError, "EP rank 0.*no endpoint"):
            self.prepare(native)
        native.get_shmem_bootstrap.return_value = (-1, [1])
        with self.assertRaisesRegex(RuntimeError, "different group"):
            self.prepare(native)

    def test_mixed_active_sessions_rejected_before_new_uid(self):
        native = Mock()
        native.get_shmem_bootstrap.return_value = (0, [])

        def gather(group, comm, size, value):
            return [value, (value[0], 1, [9])]

        with self.assertRaisesRegex(RuntimeError, "inconsistent active"):
            self.prepare(native, gather)
        native.get_shmem_unique_id.assert_not_called()

    def test_mismatched_options_rejected_before_uid_creation(self):
        native = Mock()
        native.get_shmem_bootstrap.return_value = (0, [])

        def gather(group, comm, size, value):
            return [value, ((128, *value[0][1:]), 0, [])]

        with self.assertRaisesRegex(ValueError, "all EP ranks"):
            self.prepare(native, gather)
        native.get_shmem_unique_id.assert_not_called()

    def test_mismatched_token_capacity_rejected_before_uid_creation(self):
        native = Mock()
        native.get_shmem_bootstrap.return_value = (0, [])

        def gather(group, comm, size, value):
            options = value[0]
            return [value, ((*options[:-2], 129, options[-1]), 0, [])]

        with self.assertRaisesRegex(ValueError, "all EP ranks"):
            self.prepare(native, gather)
        native.get_shmem_unique_id.assert_not_called()

    def test_mismatched_schedules_rejected_before_uid_creation(self):
        native = Mock()
        native.get_shmem_bootstrap.return_value = (0, [])

        def gather(group, comm, size, value):
            return [value, ((*value[0][:-1], "token"), 0, [])]

        with self.assertRaisesRegex(ValueError, "all EP ranks"):
            self.prepare(native, gather)
        native.get_shmem_unique_id.assert_not_called()


class BootstrapEntryPointTest(unittest.TestCase):
    def test_layout_and_notify_use_real_group_and_explicit_endpoint(self):
        import torch
        import test_dispatch_layout
        import test_notify_dispatch

        # Stop at Buffer construction: the test checks bootstrap integration
        # and cleanup, without allocating device tensors or submitting kernels.
        class StopBeforeNative(Exception):
            pass

        args = SimpleNamespace(
            ep_size=2,
            first_device=0,
            experts_per_rank=2,
            num_tokens=3,
            num_topk=2,
            num_aiv=2,
            expert_alignment=1,
            seed=2026,
            num_bytes=64 * 1024**2,
            shmem_ip_port="tcp://127.0.0.1:30001",
            dist_init_method="tcp://127.0.0.1:30002",
        )
        for module in (test_dispatch_layout, test_notify_dispatch):
            group = object()
            distributed = SimpleNamespace(
                group=SimpleNamespace(WORLD=group),
                init_process_group=Mock(),
                is_initialized=lambda: True,
                destroy_process_group=Mock(),
            )
            with (
                self.subTest(module=module.__name__),
                patch.dict(os.environ, {"ASCEND_DEEPEP_SHMEM_IP_PORT": "tcp://127.0.0.1:30000"}),
                patch.object(torch, "npu", SimpleNamespace(set_device=lambda _: None), create=True),
                patch.object(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor),
                patch.object(module, "dist", distributed),
                patch.object(module, "Buffer", side_effect=StopBeforeNative) as construct,
            ):
                with self.assertRaises(StopBeforeNative):
                    module.run_rank(0, args)
                distributed.init_process_group.assert_called_once_with(
                    backend="gloo", init_method=args.dist_init_method, rank=0, world_size=2
                )
                construct.assert_called_once_with(
                    group,
                    num_nvl_bytes=args.num_bytes,
                    explicitly_destroy=True,
                    shmem_ip_port=args.shmem_ip_port,
                )
                distributed.destroy_process_group.assert_called_once_with()
                self.assertEqual(os.environ["ASCEND_DEEPEP_SHMEM_IP_PORT"], "tcp://127.0.0.1:30000")


if __name__ == "__main__":
    unittest.main()
