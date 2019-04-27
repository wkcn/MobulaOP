#pragma once

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

template <typename ElemType, typename STREAM>
class DFA {
 public:
  DFA() { Reset(); }
  void Reset() {
    buffer.clear();
    blocks.clear();
    cur_node = N_START;
  }
  DFA& operator<<(const STREAM& stream) {
    int si = 0;
    while (si < stream.size()) {
      const ElemType& c = stream[si];
      auto& pairs = graphs[cur_node];
      bool failed = true;
      for (auto& p : pairs) {
        if ((p.first)(c)) {
          NodeType& next_node = p.second;
          switch (next_node) {
            case N_RESERVE_TO_START:
              StreamAppend(buffer, c);
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
              StreamAppend(buffer, c);
              ++si;
              cur_node = next_node;
          };
          failed = false;
          break;
        }
      }
      if (failed) {
        throw stream;
      }
    }
    return *this;
  }

 private:
  void AddBlock() {
    if (buffer.empty()) return;
    blocks.push_back({block_types[cur_node], buffer});
    buffer.clear();
  }

 protected:
  enum Node {
    N_START = 0,
    N_READY_TO_START,    // hello+
    N_RESERVE_TO_START,  // block comment
    N_SKIP_TO_START,     // hello[space]
  };
  using FType = std::function<bool(const ElemType&)>;
  using NodeType = int;
  using BlockType = int;
  std::vector<std::vector<std::pair<FType, NodeType>>> graphs;
  std::vector<BlockType> block_types;

 protected:
  virtual void StreamAppend(STREAM&, const ElemType&) = 0;

 private:
  NodeType cur_node;
  STREAM buffer;

 public:
  std::vector<std::pair<int, STREAM>> blocks;
};

class Parser2;

class Parser : public DFA<char, std::string> {
  friend class Parser2;

 public:
  Parser() : DFA() {
    graphs.resize(NUM_OF_NODES);
    FType is_any = [](const char&) { return true; };
    FType is_alnum = [](const char& c) { return isalnum(c); };
    FType is_lslash = [](const char& c) { return c == '/'; };
    FType is_rslash = [](const char& c) { return c == '\\'; };
    FType is_space = [](const char& c) {
      return c == ' ' || c == '\b' || c == '\t' || c == '\r';
    };
    FType is_newline = [](const char& c) { return c == '\n'; };
    FType is_star = [](const char& c) { return c == '*'; };
    FType is_underline = [](const char& c) { return c == '_'; };

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
    graphs[N_ANNOTATION_BLOCK].push_back(
        {is_star, N_ANNOTATION_BLOCK_READY_TO_EXIT});
    graphs[N_ANNOTATION_BLOCK].push_back({is_any, N_ANNOTATION_BLOCK});

    // N_ANNOTATION_BLOCK_READY_TO_EXIT
    graphs[N_ANNOTATION_BLOCK_READY_TO_EXIT].push_back(
        {is_lslash, N_RESERVE_TO_START});
    graphs[N_ANNOTATION_BLOCK_READY_TO_EXIT].push_back(
        {is_any, N_ANNOTATION_BLOCK});

    // N_ESCAPE
    graphs[N_ESCAPE].push_back({is_any, N_START});

    // N_SIGN
    graphs[N_SIGN].push_back({is_any, N_READY_TO_START});

    // BLOCK TYPE
    block_types.resize(NUM_OF_NODES, T_NULL);
    block_types[N_IDENTIFIER] = T_INDENTIFIER;
    block_types[N_ANNOTATION_LINE] = T_COMMENT;
    block_types[N_ANNOTATION_BLOCK] = T_COMMENT;
    block_types[N_ESCAPE] = T_ESCAPE;
    block_types[N_SIGN] = T_SIGN;
  }

 protected:
  void StreamAppend(std::string& str, const char& c) { str += c; }

 private:
  enum Node {
    N_START = 0,
    N_READY_TO_START,    // hello+
    N_RESERVE_TO_START,  // block comment
    N_SKIP_TO_START,     // hello[space]

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
    T_NULL,
    T_INDENTIFIER,
    T_ESCAPE,
    T_SIGN,
    T_COMMENT,
    NUM_OF_TYPES
  };
};

using ElemType = std::pair<int, std::string>;
class Parser2 : public DFA<ElemType, std::vector<ElemType>> {
 public:
  Parser2() : DFA() {
    graphs.resize(NUM_OF_NODES);
    FType is_any = [](const ElemType&) { return true; };
    FType is_mobula_head = [](const ElemType& e) {
      if (e.first != Parser::T_INDENTIFIER) return false;
      return e.second == "MOBULA_KERNEL";
    };
    FType is_lbracket = [](const ElemType& e) {
      if (e.first != Parser::T_SIGN) return false;
      return e.second == "(";
    };
    FType is_rbracket = [](const ElemType& e) {
      if (e.first != Parser::T_SIGN) return false;
      return e.second == ")";
    };
    FType is_identifier = [](const ElemType& e) {
      return e.first == Parser::T_INDENTIFIER;
    };

    graphs[N_START].push_back({is_mobula_head, N_HEAD});
    graphs[N_START].push_back({is_any, N_SKIP_TO_START});

    graphs[N_HEAD].push_back({is_identifier, N_FUNC_NAME});
    graphs[N_FUNC_NAME].push_back({is_lbracket, N_ARGS});

    graphs[N_ARGS].push_back({is_rbracket, N_RESERVE_TO_START});
    graphs[N_ARGS].push_back({is_any, N_ARGS});

    block_types.resize(NUM_OF_NODES, T_NULL);
  }

 protected:
  void StreamAppend(std::vector<ElemType>& str, const ElemType& c) {
    str.push_back(c);
  }

 private:
  enum Node {
    N_START = 0,
    N_READY_TO_START,    // hello+
    N_RESERVE_TO_START,  // block comment
    N_SKIP_TO_START,     // hello[space]

    N_HEAD,
    N_FUNC_NAME,
    N_ARGS,

    NUM_OF_NODES
  };
  enum Type { T_NULL, NUM_OF_TYPES };
};
