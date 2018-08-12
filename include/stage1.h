#ifndef _H_STAGE_1
#define _H_STAGE_1

#include <buffer.h>
#include <string>
#include <fstream>
#include <vector>
#include "options.h"

/**
*  I/O worker
*/

template <class T>
class Stage1
{
    Buffer<T>&  buffer_;
    const std::vector<std::string> fnames;
    int jobs, passes;
public:
    Stage1(Buffer<T>& buffer, const Options& opts);
    void run();
};

#endif
