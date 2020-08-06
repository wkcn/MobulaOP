struct MyStruct {
  int hello;
  float mobula;
};

MOBULA_FUNC float hello(MyStruct *hi) {
  LOG(INFO) << "Hello Mobula: " << hi->hello << ", " << hi->mobula;
  return hi->hello + hi->mobula;
}
