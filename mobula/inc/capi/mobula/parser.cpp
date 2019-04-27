#include "parser.h"
#include <fstream>

int main() {
  Parser parser;
  std::ifstream fin("code.cpp");
  std::string buffer;
  getline(fin, buffer);
  std::cout << "-----------------------------" << std::endl;
  while (!fin.eof()) {
    parser << buffer << "\n";
    getline(fin, buffer);
  }
  std::cout << "*****************************" << std::endl;
  int i = 0;
  for (auto &p : parser.blocks) {
    std::cout << i++ << " = " << p.first << "-" << p.second << std::endl;
  }
  std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;

  Parser2 parser2;
  parser2 << parser.blocks;
  for (auto &p : parser2.blocks) {
    for (auto &e : p.second) {
      std::cout << e.second << std::endl;
    }
    std::cout << std::endl;
  }
  return 0;
}
