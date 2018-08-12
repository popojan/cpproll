#include "stage1.h"
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

extern volatile bool interrupted;

template <class T> Stage1<T>::Stage1(
    Buffer<T>& buffer,
    const Options& opts
) : buffer_(buffer), fnames(opts.files), jobs(opts.jobs), passes(opts.passes) {

}

template <class T> void Stage1<T>::run() {
    std::string line;
    for(int j = 0; j < passes && !interrupted; ++j) {
        for(auto fit = fnames.begin(); fit != fnames.end() && !interrupted; ++fit) {
            auto fname(*fit);
            if(fname.back() == 'z' && fname[fname.size()-2] == 'g' && fname[fname.size()-3] == '.') {
                std::ifstream file(fname, std::ios_base::in | std::ios_base::binary);
                boost::iostreams::filtering_istream fin;
                fin.push(boost::iostreams::gzip_decompressor());
                fin.push(file);
                while(std::getline(fin, line)) {
                    if(interrupted) break;
                    buffer_.add(line);
                }
            }
            else {
                std::ifstream fin(fname);
                while(std::getline(fin, line)) {
                    if(interrupted) break;
                    buffer_.add(line);
                }
            }
        }
    }
    if(interrupted)
        buffer_.clear();
    for(int i = 0; i < jobs; ++i)
        buffer_.add(T());
}

template class Stage1<std::string>;
