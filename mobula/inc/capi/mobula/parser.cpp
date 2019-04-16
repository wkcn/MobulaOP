#include "parser.h"
#include <fstream>

std::vector<std::pair<Parser::FType, Parser::Node> > Parser::graphs[NUM_OF_NODES];

int main() {
  Parser::StaticInit();
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
    std::cout << i++ << " = " << p.second << std::endl;
  }
  std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
  return 0;
}
