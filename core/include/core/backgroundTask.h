#pragma once

#include <core/defines.h>
#include <core/diagnostics.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace pts {

class TaskProgress {
   public:
    void set_progress(float value) {
        m_progress.store(value, std::memory_order_relaxed);
    }

    void set_status(const std::string& status) {
        std::lock_guard lock(m_mutex);
        m_status = status;
    }

    float progress() const {
        return m_progress.load(std::memory_order_relaxed);
    }

    std::string status() const {
        std::lock_guard lock(m_mutex);
        return m_status;
    }

   private:
    std::atomic<float> m_progress{0.0f};
    mutable std::mutex m_mutex;
    std::string m_status;
};

template <typename T>
class BackgroundTask {
   public:
    using WorkFn = std::function<T(TaskProgress&)>;

    NO_COPY(BackgroundTask);

    BackgroundTask(std::string name, WorkFn work) : m_name(std::move(name)) {
        PRECONDITION(work);
        m_thread = std::thread([this, work = std::move(work)]() {
            T result = work(m_progress);
            {
                std::lock_guard lock(m_result_mutex);
                m_result.emplace(std::move(result));
            }
            m_done.store(true, std::memory_order_release);
        });
    }

    BackgroundTask(BackgroundTask&& other) noexcept
        : m_name(std::move(other.m_name)),
          m_thread(std::move(other.m_thread)),
          m_done(other.m_done.load(std::memory_order_acquire)),
          m_result(std::move(other.m_result)) {
        other.m_done.store(false, std::memory_order_relaxed);
    }

    BackgroundTask& operator=(BackgroundTask&& other) noexcept {
        PRECONDITION_MSG(!m_thread.joinable(),
                         "Cannot move-assign over a BackgroundTask with a running thread");
        m_name = std::move(other.m_name);
        m_thread = std::move(other.m_thread);
        m_done.store(other.m_done.load(std::memory_order_acquire), std::memory_order_release);
        other.m_done.store(false, std::memory_order_relaxed);
        {
            std::lock_guard lock(other.m_result_mutex);
            m_result = std::move(other.m_result);
        }
        return *this;
    }

    ~BackgroundTask() {
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }

    bool is_done() const {
        return m_done.load(std::memory_order_acquire);
    }

    float progress() const {
        return m_progress.progress();
    }

    std::string status() const {
        return m_progress.status();
    }

    const std::string& name() const {
        return m_name;
    }

    T take_result() {
        PRECONDITION_MSG(is_done(), "Cannot take result before task is done");
        if (m_thread.joinable()) {
            m_thread.join();
        }
        std::lock_guard lock(m_result_mutex);
        INVARIANT_MSG(m_result.has_value(), "Result already taken");
        T result = std::move(*m_result);
        m_result.reset();
        return result;
    }

   private:
    std::string m_name;
    TaskProgress m_progress;
    std::thread m_thread;
    std::atomic<bool> m_done{false};
    std::mutex m_result_mutex;
    std::optional<T> m_result;
};

}  // namespace pts
