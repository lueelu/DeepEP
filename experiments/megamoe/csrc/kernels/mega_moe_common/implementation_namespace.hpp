// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

// Compile each backend with its original namespace and timer symbols.
// The selector is a build-target definition, never a device runtime input.
#pragma once

#if defined(DEEPEP_MEGAMOE_URMA)
#define DEEPEP_MEGAMOE_IMPL_NAMESPACE UrmaMegaMoeImpl
#define DEEPEP_MEGAMOE_TIMER_NAMESPACE UrmaAscendTimer
#define DEEPEP_MEGAMOE_TIMER_DEVICE UrmaAscendTimerDevice
#define DEEPEP_MEGAMOE_TIMER_TILE_SCOPE UrmaAscendTimerTileScope
#define DEEPEP_MEGAMOE_TIMER_SPAN_SCOPE UrmaAscendTimerSpanScope
#define DEEPEP_MEGAMOE_TIMER_WAIT_SCOPE UrmaAscendTimerWaitScope

#else
#define DEEPEP_MEGAMOE_IMPL_NAMESPACE MegaMoeImpl
#define DEEPEP_MEGAMOE_TIMER_NAMESPACE AscendTimer
#define DEEPEP_MEGAMOE_TIMER_DEVICE AscendTimerDevice
#define DEEPEP_MEGAMOE_TIMER_TILE_SCOPE AscendTimerTileScope
#define DEEPEP_MEGAMOE_TIMER_SPAN_SCOPE AscendTimerSpanScope
#define DEEPEP_MEGAMOE_TIMER_WAIT_SCOPE AscendTimerWaitScope

#endif
