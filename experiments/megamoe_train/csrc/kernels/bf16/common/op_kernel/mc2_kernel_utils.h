// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026
// Modified by guofanfeng 2026

/*!
 * \file mc2_kernel_utils.h
 * \brief
 */

#ifndef MC2_KERNEL_UTILS_H
#define MC2_KERNEL_UTILS_H

namespace AscendC {

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

}  // namespace AscendC
#endif  // MC2_KERNEL_UTILS_H
