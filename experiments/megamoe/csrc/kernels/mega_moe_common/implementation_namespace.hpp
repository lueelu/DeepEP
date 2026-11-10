// Compile each backend with its original namespace and timer symbols.
// The selector is a build-target definition, never a device runtime input.
#pragma once

#if defined(DEEPEP_MEGAMOE_URMA)
#define DEEPEP_MEGAMOE_IMPL_NAMESPACE UbxMegaMoeImpl
#define DEEPEP_MEGAMOE_TIMER_NAMESPACE UbxAscendTimer
#define DEEPEP_MEGAMOE_TIMER_DEVICE UbxAscendTimerDevice
#define DEEPEP_MEGAMOE_TIMER_TILE_SCOPE UbxAscendTimerTileScope
#define DEEPEP_MEGAMOE_TIMER_SPAN_SCOPE UbxAscendTimerSpanScope
#define DEEPEP_MEGAMOE_TIMER_WAIT_SCOPE UbxAscendTimerWaitScope

#else
#define DEEPEP_MEGAMOE_IMPL_NAMESPACE MegaMoeImpl
#define DEEPEP_MEGAMOE_TIMER_NAMESPACE AscendTimer
#define DEEPEP_MEGAMOE_TIMER_DEVICE AscendTimerDevice
#define DEEPEP_MEGAMOE_TIMER_TILE_SCOPE AscendTimerTileScope
#define DEEPEP_MEGAMOE_TIMER_SPAN_SCOPE AscendTimerSpanScope
#define DEEPEP_MEGAMOE_TIMER_WAIT_SCOPE AscendTimerWaitScope

#endif
