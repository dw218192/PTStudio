#pragma once

#include <boost/signals2/signal.hpp>

namespace pts {
template <typename Signature>
using Signal = boost::signals2::signal<Signature>;
}  // namespace pts
