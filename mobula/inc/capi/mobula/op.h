#pragma once

#include <iostream>
#include <string>
#include <fstream>

namespace mobula {
namespace op {
  int load(const std::string path) {
    std::string fname = path.substr(path.rfind('/') + 1) + ".cpp";
    std::string fulpath = path + '/' + fname;
    std::ifstream fin(fulpath);
    if (fin.fail()) {
      std::cout << fulpath << std::endl;
      return -1;
    }
    std::string buf;
    std::getline(fin, buf);
    enum ParserState {
      kNone,
      kHead,
      kName,
      kPars,
    };
    ParserState state = kNone;
    while(!fin.eof()) {
      if (state == kNone) {
        // Whether MOBULA_KERNEL exists?
        // trim the first space
        buf.erase(0, buf.find_first_not_of(" "));
        std::string head = buf.substr(0, buf.find(' '));
        if (head == "MOBULA_KERNEL") {
          std::cout << "YES" << std::endl;
          std::cout << buf << std::endl;
        }
      }
      // std::cout << buf << std::endl;
      std::getline(fin, buf);
    }
    return 0;
  }
}
}
