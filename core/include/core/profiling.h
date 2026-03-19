#pragma once

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>

#if defined(TRACY_DELAYED_INIT) && defined(TRACY_MANUAL_LIFETIME)
#include <client/TracyProfiler.hpp>
// With manual lifetime, there is no static s_profiler and no atexit handler.
// We start the profiler explicitly but skip shutdown — Tracy 0.13.1's
// ShutdownProfiler() deadlocks because Worker() blocks in Accept() with no
// timeout.  The OS reclaims all threads, sockets, and memory at process exit.
#define PTS_STARTUP_PROFILER() tracy::StartupProfiler()
#define PTS_SHUTDOWN_PROFILER()
#else
#define PTS_STARTUP_PROFILER()
#define PTS_SHUTDOWN_PROFILER()
#endif

#define PTS_ZONE_SCOPED ZoneScoped
#define PTS_ZONE_NAMED(name) ZoneScopedN(name)
#define PTS_FRAME_MARK FrameMark
#else
#define PTS_STARTUP_PROFILER()
#define PTS_SHUTDOWN_PROFILER()
#define PTS_ZONE_SCOPED
#define PTS_ZONE_NAMED(name)
#define PTS_FRAME_MARK
#endif
