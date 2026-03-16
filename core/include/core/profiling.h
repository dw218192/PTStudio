#pragma once

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#define PTS_ZONE_SCOPED ZoneScoped
#define PTS_ZONE_NAMED(name) ZoneScopedN(name)
#define PTS_FRAME_MARK FrameMark
#else
#define PTS_ZONE_SCOPED
#define PTS_ZONE_NAMED(name)
#define PTS_FRAME_MARK
#endif
