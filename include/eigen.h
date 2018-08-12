#ifndef EIGEN_CONFIG_H_
#define EIGEN_CONFIG_H_

#include <boost/serialization/array.hpp>

#define EIGEN_DENSEBASE_PLUGIN "serialize.h"

#include <Eigen/Core>

#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

/// Boost Serialization Helper

template <typename T>
bool serialize(const T& data, const std::string& filename) {
  std::ofstream ofs(filename.c_str(), std::ios::out);
  if (!ofs.is_open())
    return false;
  {
    boost::archive::binary_oarchive oa(ofs);
    oa << data;
  }
  ofs.close();
  return true;
}

template <typename T>
bool deSerialize(T& data, const std::string& filename) {
  std::ifstream ifs(filename.c_str(), std::ios::in);
  if (!ifs.is_open())
    return false;
  {
    boost::archive::binary_iarchive ia(ifs);
    ia >> data;
  }
  ifs.close();
  return true;
}

#endif
