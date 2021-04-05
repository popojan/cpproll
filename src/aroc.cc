// Copyright (c) 2014 vassilis@entropiece.com
// www.entropiece.com

// cppAROC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "aroc.h"

AROC::AROC(int m): bins(m), data(bins+1, {0, 0}) {

}

void AROC::push(int gold, float pred) {
    size_t j = bins - static_cast<size_t>(std::floor(pred * bins));
    if(gold > 0) {
        data[j].first += 1;
    } else if (gold <= 0) {
        data[j].second += 1;
    }
}

void AROC::run() {
    getAROCFromData(this->data);
}

void AROC::clear() {
    data.clear();
    TP.clear();
    FP.clear();
}

void AROC::readFile(std::string filename){
  float score;
  int label;
  std::ifstream inFile(filename.c_str());

  if(!inFile) {
    std::cerr << "File "<< filename << " not found.";
  }

  // read the data from the file
  while(!inFile.eof()) {
    inFile  >>  score >> label;
    if (!inFile) break;
    this->data.push_back(std::make_pair(score,label));
  }
  getAROCFromData(this->data);
}

AROC::~AROC() {
}

void AROC::writeToFile(std::string filename){
  std::ofstream outfile(filename.c_str());
  for (size_t i = 0; i < this->TP.size(); i++) {
    outfile  <<  FP[i] << "\t" << TP[i] << std::endl;
  }
  std::cout <<  "AROC curve points saved to file " << filename << std::endl;
}

float AROC::get95ErrorRate(){
  // TODO - Implement this
  return 1.0;
}

float AROC::getAreaUnderCurve(){
  // TODO - Implement this
  size_t size = TP.size();
  float q1,q2,p1,p2;
  q1 = FP[0];
  q2 = TP[0];
  float area = 0.0;
  for(size_t i=1;i < size;++i){
    p1 = FP[i];
    p2 = TP[i];
    area += sqrt(pow( ((1-q1)+(1-p1))/2 * (q2-p2),2));
    q1=p1;
    q2=p2;   
  }
  return area;
}

bool compare(const std::pair<float,int>&i, const std::pair<float,int>&j){
  return i.first > j.first;
}

// Main AROC algorithm from the paper
// AROC Graphs: Notes and Practical Considerations for
// Researchers - Tom Fawcett (Algorithm 2)
// @TECHREPORT{Fawcett04rocgraphs:,
//     author = {Tom Fawcett},
//     title = {AROC Graphs: Notes and Practical Considerations for Researchers},
//     year = {2004}
// }
void AROC::getAROCFromData(std::vector<std::pair<size_t,size_t> > data){
  this->TP.clear();
  this->FP.clear();
  size_t L = data.size();
  size_t P = 0;
  size_t N = 0;

  // count positive and negative class occurences
  for (size_t j = 0; j < data.size(); j++) {
	P+= data[j].first;
	N+= data[j].second;
  }
  //init FP TP counters
  double FP = 0;
  double TP = 0;
  std::vector<std::pair<float,float> > R;
  //loop through all data
  for (size_t i = 0; i < L; i++) {
    //f_i = data[i].first;
    size_t p = data[i].first;
    size_t n = data[i].second;
    if(p > 0 || n > 0) {
        this->TP.push_back(TP/P);
        this->FP.push_back(FP/N);
    }
	TP = TP + p;
	FP = FP + n;
  }
  //add point 1-1
  this->TP.push_back(TP/P);
  this->FP.push_back(FP/N);
}


