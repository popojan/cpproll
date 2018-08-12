#ifndef _H_HASH
#define _H_HASH

#include "murmurhash3.hpp"
#include <string>

template <bool Use128Mode = true>
struct murmur3 {
    typedef   boost::bloom_filters::detail::murmurhash3_dispatch<boost::bloom_filters::detail::Is64Bit::value, Use128Mode> my_dispatch_type;
    explicit murmur3(size_t seed = 0): Seed(seed) {}
    size_t operator()(const std::string& t) {
        static my_dispatch_type dispatcher;
        size_t out[2] = {0,0};
        dispatcher(reinterpret_cast<const void*>(t.c_str()), t.size(), Seed, &out);
        return out[0];
    }
private:
    const size_t Seed;
};

#endif
