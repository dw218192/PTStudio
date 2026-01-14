#pragma once
#include <functional>

#include "utils.h"
namespace PTS {
struct TransactionScope {
    NO_COPY_MOVE(TransactionScope);

    template <typename F>
    TransactionScope(F&& f) noexcept : m_f(std::forward<F>(f)) {
    }
    ~TransactionScope() noexcept {
        if (!m_commit) {
            m_f();
        }
    }
    void commit() noexcept {
        m_commit = true;
    }

   private:
    std::function<void()> m_f;
    bool m_commit{false};
};
}  // namespace PTS