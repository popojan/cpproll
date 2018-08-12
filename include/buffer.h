#ifndef _H_BUFFER_1
#define _H_BUFFER_1

#include <deque>
#include <mutex>
#include <condition_variable>

template <class T>
class Buffer
{
public:
    void add(const T& num) {
        while (true) {
            std::unique_lock<std::mutex> locker(mu);
            cond.wait(locker, [this](){return buffer_.size() < size_;});
            buffer_.push_back(num);
            locker.unlock();
            cond.notify_all();
            return;
        }
    }
    T remove() {
        while (true)
        {
            std::unique_lock<std::mutex> locker(mu);
            cond.wait(locker, [this](){return buffer_.size() > 0;});
            T back = buffer_.front();
            buffer_.pop_front();
            locker.unlock();
            cond.notify_all();
            return back;
        }
    }
    void clear() {
        while (true)
        {
            std::unique_lock<std::mutex> locker(mu);
            buffer_.clear();
            locker.unlock();
            cond.notify_all();
            return;
        }
    }
    Buffer() {}
private:
   // Add them as member variables here
    std::mutex mu;
    std::condition_variable cond;

   // Your normal variables here
    std::deque<T> buffer_;
    const unsigned int size_ = 128;
};

#endif
