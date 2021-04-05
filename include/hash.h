#ifndef _H_HASH
#define _H_HASH

#include "murmur3.h"
#include <string>

struct murmur3 {
    explicit murmur3(size_t seed = 0): Seed(seed) {}
    size_t operator()(const std::string& t) {
        size_t out[2] = {0, 0};
        MurmurHash3_x64_128(t.c_str(), t.length(), Seed, out);
        return out[0];
    }
private:
    const size_t Seed;
};


#endif
