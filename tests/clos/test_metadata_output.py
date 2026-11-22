"""Public cached-dispatch opt-in forwarding, without a device runtime."""

import pytest

from tests.api.test_clos_compatibility import run_host


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("option", [None, "CACHED_WEIGHT_PACK", "CACHED_OMIT_TOPK_OUTPUT"])
def test_cached_metadata_keeps_route_ownership_and_expansion_consumers(expanded, option):
    run_host(f"""
        import os
        from contextlib import nullcontext
        from deep_ep.backends.direct import elastic as m
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        class Submitted(Exception): pass
        class Tensor:
            def __init__(self, shape, dtype='torch.bfloat16', device='npu'):
                self.shape,self.dtype,self.device=shape,dtype,device
            def size(self, dim): return self.shape[dim]
            def element_size(self): return 2
        torch.empty=lambda shape,**kw:Tensor(shape,**kw)
        for key in ('CACHED_WEIGHT_PACK','CACHED_OMIT_TOPK_OUTPUT'):
            os.environ['DEEPEP_CLOS_'+key] = str(int(key=={option!r}))
        for name in ('_validate_dispatch_scope','_validate_dispatch_tensors'):
            setattr(m,name,lambda *a:None)
        m._get_allocation_stream_id=lambda _:0
        m._try_record_event=lambda _:None
        m._current_npu_stream=lambda:'caller'
        m._use_stream_after_event=lambda *a:nullcontext()
        m._use_comm_stream=lambda *a:nullcontext()
        m.load_native_module=lambda:SimpleNamespace(dispatch_metadata_output_version=1)
        x,idx,weights=Tensor((65,2816)),Tensor((65,8)),Tensor((65,8))
        head,plan=object(),object()
        def submit(*args):
            assert args[2] is idx and args[3] is weights
            if {option is not None}:
                assert args[-2] is head and args[-1] is {not expanded}
                assert args[-4] is plan
            else:
                assert args[-2] is plan
            raise Submitted
        policy=resolve_udma_policy(True,8,'udma',None,None,None,None,None)
        obj=SimpleNamespace(_ensure_alive=lambda:None,_validate_handle=lambda _:None,
            use_fp8_dispatch=False,num_max_tokens_per_rank=65,num_ranks=8,hidden=2816,num_topk=8,
            get_buffer_size_hint=lambda *a:0,group=None,allow_hybrid_mode=True,
            num_bytes=1,num_device_aiv=64,_udma_auto_policy=policy,transport='udma',
            _communication_stream='comm',_use_single_source_combine=False,
            _native=SimpleNamespace(intranode_dispatch_with_plan=submit))
        handle=SimpleNamespace(num_sms=8,num_experts=256,num_max_tokens_per_rank=65,
            expert_alignment=1,do_expand={expanded},topk_idx=idx,num_recv_tokens=0,
            _dispatch_layout=(object(),object(),None,None,Tensor((65,8)),head),_dispatch_plan=plan)
        try:
            m.ElasticBuffer.dispatch(obj,x,handle=handle,topk_weights=weights,num_sms=8,do_expand={expanded})
        except Submitted: pass
        else: raise AssertionError('missing native dispatch')
    """)
