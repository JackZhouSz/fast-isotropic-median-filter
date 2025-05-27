// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FAST_ISOTROPIC_MEDIAN_FILTER_THREAD_POOL_H_
#define FAST_ISOTROPIC_MEDIAN_FILTER_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace fast_isotropic_median_filter {

class ThreadPool {
 public:
  // Creates a thread pool with the specified number of threads.
  explicit ThreadPool(int num_threads = DefaultNumThreads())
      : active_tasks_(0), num_threads_(num_threads) {
    for (int i = 0; i < num_threads; i++) {
      threads_.push_back(std::thread([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
              return;  // Exits the thread.
            }
            task = std::move(tasks_.front());
            tasks_.pop_front();
          }
          task();
          {
            std::unique_lock<std::mutex> lock(mutex_);
            --active_tasks_;
            cv_.notify_all();
          }
        }
      }));
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (stop_) {
        return;
      }
      stop_ = true;
    }
    cv_.notify_all();
    for (std::thread& thread : threads_) {
      thread.join();
    }
  }

  // Adds a task to the thread pool.
  void Enqueue(std::function<void()> fn) {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.push_back(fn);
    ++active_tasks_;
    cv_.notify_one();
  }

  // Barrier to wait until all queued tasks are completed.
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return active_tasks_ == 0 && tasks_.empty(); });
  }

  static int DefaultNumThreads() { return std::thread::hardware_concurrency(); }

  int NumThreads() const { return num_threads_; }

 private:
  bool stop_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<std::thread> threads_;
  std::deque<std::function<void()>> tasks_;
  std::atomic<int> active_tasks_;
  int num_threads_;
};

}  // namespace fast_isotropic_median_filter

#endif  // FAST_ISOTROPIC_MEDIAN_FILTER_THREAD_POOL_H_
