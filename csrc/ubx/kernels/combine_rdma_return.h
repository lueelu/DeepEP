// Copyright (c) 2026, Lu Lu

#pragma once

#include "combine_return_schedule.h"
#include "combine_transfer_batch.h"
namespace UbxCombine {
using ReturnBatch = TransferBatch;
// Caller has packed this output and acquired V->MTE3.
__aicore__ inline void SendReturnPacket(ReturnBatch& batch, const __gm__ UbxDispatch::CommArgs* args, uint32_t peer,
                                        uint64_t offset, AscendC::LocalTensor<uint8_t> output, uint32_t bytes)
{
    if (UbxTransport::RemoteWrite(args, peer))
        batch.Enqueue(peer, offset, output);
    else {
        UbxTransport::Check(args, peer, offset, bytes);
        AscendC::DataCopy(UbxTransport::Tensor(UbxTransport::Address(args, peer, offset), false), output, bytes);
    }
}
}  // namespace UbxCombine
