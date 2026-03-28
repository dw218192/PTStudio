#pragma once

#include <core/defines.h>
#include <core/diagnostics.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <variant>

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

    void reset() {
        m_progress.store(0.0f, std::memory_order_relaxed);
        std::lock_guard lock(m_mutex);
        m_status.clear();
    }

   private:
    std::atomic<float> m_progress{0.0f};
    mutable std::mutex m_mutex;
    std::string m_status;
};

/// Persistent worker that processes jobs on a dedicated TBB arena thread.
/// Latest-wins semantics: submitting a new job replaces any pending (unstarted) job.
template <typename Job, typename Result>
class Worker {
   public:
    using WorkFn = std::function<Result(Job&&, TaskProgress&)>;

    NO_COPY_MOVE(Worker);

    explicit Worker(WorkFn work) : m_work_fn(std::move(work)), m_arena(2, 1) {
        PRECONDITION(m_work_fn);
        m_arena.execute([this] { m_group.run([this] { loop(); }); });
    }

    ~Worker() {
        shutdown();
    }

    /// Submit a job. Replaces any pending (not yet started) job.
    void submit(Job job) {
        {
            std::lock_guard lock(m_mutex);
            PRECONDITION_MSG(!m_shutdown, "Cannot submit to a shut-down worker");
            m_pending.emplace(std::move(job));
        }
        m_cv.notify_one();
    }

    /// Lockfree check: is a result available?
    bool has_result() const {
        return m_result.load(std::memory_order_acquire) != nullptr;
    }

    /// Take the result (caller owns the pointer contents). Returns nullopt if none ready.
    std::optional<Result> take_result() {
        auto* ptr = m_result.exchange(nullptr, std::memory_order_acq_rel);
        if (!ptr) return std::nullopt;
        Result r(std::move(*ptr));
        delete ptr;
        return r;
    }

    /// Current task progress (valid while a job is running).
    float progress() const {
        return m_progress.progress();
    }
    std::string status() const {
        return m_progress.status();
    }

    /// Signal shutdown and block until in-flight work completes.
    void shutdown() {
        {
            std::lock_guard lock(m_mutex);
            if (m_shutdown) return;
            m_shutdown = true;
        }
        m_cv.notify_one();
        m_arena.execute([this] { m_group.wait(); });
        // Clean up any unconsumed result
        delete m_result.exchange(nullptr, std::memory_order_acq_rel);
    }

   private:
    void loop() {
        while (true) {
            std::optional<Job> job_local;
            {
                std::unique_lock lock(m_mutex);
                m_cv.wait(lock, [&] { return m_pending.has_value() || m_shutdown; });
                if (m_shutdown && !m_pending.has_value()) return;
                job_local.emplace(std::move(*m_pending));
                m_pending.reset();
            }
            m_progress.reset();
            Result r = m_work_fn(std::move(*job_local), m_progress);
            delete m_result.exchange(new Result(std::move(r)), std::memory_order_acq_rel);
        }
    }

    WorkFn m_work_fn;
    tbb::task_arena m_arena;
    tbb::task_group m_group;

    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::optional<Job> m_pending;
    bool m_shutdown{false};

    TaskProgress m_progress;
    std::atomic<Result*> m_result{nullptr};
};

/// One-shot async task that runs a single function and stores the result.
template <typename T>
class OneShotTask {
   public:
    using WorkFn = std::function<T(TaskProgress&)>;

    NO_COPY_MOVE(OneShotTask);

    OneShotTask(std::string name, WorkFn work)
        : m_name(std::move(name)),
          m_worker([work = std::move(work)](std::monostate&&, TaskProgress& p) -> T {
              return work(p);
          }) {
        m_worker.submit(std::monostate{});
    }

    bool is_done() const {
        return m_worker.has_result();
    }

    float progress() const {
        return m_worker.progress();
    }
    std::string status() const {
        return m_worker.status();
    }
    const std::string& name() const {
        return m_name;
    }

    T take_result() {
        PRECONDITION_MSG(is_done(), "Cannot take result before task is done");
        auto opt = m_worker.take_result();
        INVARIANT_MSG(opt.has_value(), "Result already taken");
        return std::move(*opt);
    }

   private:
    std::string m_name;
    Worker<std::monostate, T> m_worker;
};

}  // namespace pts
