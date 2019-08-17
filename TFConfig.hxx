#pragma once

#include <memory>

#include <tensorflow/core/framework/tensor.h>

class TFConfig{
public:
  TFConfig()
  {
  }

  ~TFConfig()
  {
  }

  static inline const tensorflow::Scope & scope() { return scope_; }
  static inline const std::unique_ptr<tensorflow::ClientSession> & session() { return session_; }

  inline void graph(const string prefix)
  {
       GraphDef graph;
       scope_.ToGraphDef(&graph);
       SummaryWriterInterface* w;
       TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "/home/vincent/HPC/tensorboard", prefix, Env::Default(), &w));
       TF_CHECK_OK(w->WriteGraph(0, std::make_unique<GraphDef>(graph)));
  }


private:
  static inline tensorflow::Scope scope_ = tensorflow::Scope::NewRootScope();
  static inline std::unique_ptr<tensorflow::ClientSession> session_ = std::make_unique<tensorflow::ClientSession>(TFConfig::scope());
};
