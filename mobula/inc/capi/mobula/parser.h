#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <map>
#include <utility>

class Parser {
public:
  Parser() {
    Reset();
  }
  void Reset() {
    buffer.clear();
    blocks.clear();
    cur_node = N_START;
  }
  static void StaticInit() {
    FType is_any = [](const char&){return true;};
    FType is_alnum = [](const char& c){return isalnum(c);};
    FType is_lslash = [](const char& c){return c == '/';};
    FType is_rslash = [](const char& c){return c == '\\';};
    FType is_space = [](const char& c){return c == ' ' || c == '\b' || c == '\t' || c == '\r';};
    FType is_newline = [](const char& c){return c == '\n';};
    FType is_star = [](const char& c){return c == '*';};
    FType is_underline = [](const char& c){return c == '_';};

    // N_START
    graphs[N_START].push_back({is_alnum, N_IDENTIFIER});
    graphs[N_START].push_back({is_lslash, N_ANNOTATION_START});
    graphs[N_START].push_back({is_rslash, N_ESCAPE});
    graphs[N_START].push_back({is_space, N_SKIP_TO_START}); 
    graphs[N_START].push_back({is_newline, N_SKIP_TO_START});
    graphs[N_START].push_back({is_any, N_SIGN});

    // N_IDENTIFIER 
    graphs[N_IDENTIFIER].push_back({is_alnum, N_IDENTIFIER});
    graphs[N_IDENTIFIER].push_back({is_underline, N_IDENTIFIER});
    graphs[N_IDENTIFIER].push_back({is_any, N_READY_TO_START});

    // N_ANNOTATION_START
    graphs[N_ANNOTATION_START].push_back({is_lslash, N_ANNOTATION_LINE});
    graphs[N_ANNOTATION_START].push_back({is_star, N_ANNOTATION_BLOCK});

    // N_ANNOTATION_LINE
    graphs[N_ANNOTATION_LINE].push_back({is_newline, N_SKIP_TO_START});
    graphs[N_ANNOTATION_LINE].push_back({is_any, N_ANNOTATION_LINE});

    // N_ANNOTATION_BLOCK
    graphs[N_ANNOTATION_BLOCK].push_back({is_star, N_ANNOTATION_BLOCK_READY_TO_EXIT});
    graphs[N_ANNOTATION_BLOCK].push_back({is_any, N_ANNOTATION_BLOCK});

    // N_ANNOTATION_BLOCK_READY_TO_EXIT
    graphs[N_ANNOTATION_BLOCK_READY_TO_EXIT].push_back({is_lslash, N_RESERVE_TO_START});
    graphs[N_ANNOTATION_BLOCK_READY_TO_EXIT].push_back({is_any, N_ANNOTATION_BLOCK});

    // N_ESCAPE
    graphs[N_ESCAPE].push_back({is_any, N_START});

    // N_SIGN
    graphs[N_SIGN].push_back({is_any, N_READY_TO_START});
  }
  Parser& operator<<(const std::string &str) {
    int si = 0;
    while (si < str.size()) {
      const char &c = str[si];
      auto &pairs = graphs[cur_node];
      bool failed = true;
      std::cout << cur_node << ": " << c << std::endl;
      for (auto &p : pairs) {
        if ((p.first)(c)) {
          Node &next_node = p.second;
          switch (next_node) {
            case N_RESERVE_TO_START:
              buffer += c;
            case N_SKIP_TO_START:
              AddBlock();
              ++si;
              cur_node = N_START;
              break;
            case N_READY_TO_START:
              AddBlock();
              cur_node = N_START;
              break;
            default:
              buffer += c;
              ++si;
              cur_node = next_node;
          };
          failed = false;
          break;
        }
      }
      if (failed) {
        std::cout << "ERROR: " << str << std::endl;
        throw "OH NO";
      }
    }
    return *this;
  }
private:
  void AddBlock() {
    if (buffer.empty()) return;
    blocks.push_back({T_INDENTIFIER, buffer});
    buffer.clear();
  }
private:
  enum Node {
    N_START = 0,
    N_READY_TO_START, // hello+
    N_RESERVE_TO_START, // /* */
    N_SKIP_TO_START, // hello[space]
    N_IDENTIFIER,
    N_ANNOTATION_START,
    N_ANNOTATION_LINE,
    N_ANNOTATION_BLOCK,
    N_ANNOTATION_BLOCK_READY_TO_EXIT,
    N_ESCAPE,
    N_SIGN,
    NUM_OF_NODES 
  };
  enum Type {
    T_INDENTIFIER,
    T_ANNOTATION,
    T_ESCAPE,
    T_SYMBOL,
    NUM_OF_TYPES
  };
  using FType = std::function<bool(const char&)>;
  static std::vector<std::pair<FType, Node> > graphs[NUM_OF_NODES];
private:
  Node cur_node;
  std::string buffer;
public:
  std::vector<std::pair<Type, std::string> > blocks;
};
