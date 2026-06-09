#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace
{
constexpr std::size_t kObservationSize = 41;
constexpr std::size_t kActionSize = 3;

std::string default_policy_path()
{
  const char * workspace = std::getenv("AEROSTRIKE_WS");
  if (workspace == nullptr || std::string(workspace).empty()) {
    return "checkpoints/aerostrike_policy.onnx";
  }
  return std::string(workspace) + "/checkpoints/aerostrike_policy.onnx";
}

std::vector<std::string> read_io_names(Ort::Session & session, const bool read_inputs)
{
  Ort::AllocatorWithDefaultOptions allocator;
  const auto count = read_inputs ? session.GetInputCount() : session.GetOutputCount();

  std::vector<std::string> names;
  names.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    auto name = read_inputs ?
      session.GetInputNameAllocated(index, allocator) :
      session.GetOutputNameAllocated(index, allocator);
    names.emplace_back(name.get());
  }
  return names;
}

std::vector<const char *> make_name_ptrs(const std::vector<std::string> & names)
{
  std::vector<const char *> ptrs;
  ptrs.reserve(names.size());
  for (const auto & name : names) {
    ptrs.push_back(name.c_str());
  }
  return ptrs;
}
}  // namespace

int main(int argc, char ** argv)
{
  const std::string policy_path = argc > 1 ? argv[1] : default_policy_path();

  try {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "aerostrike_policy_inference_smoke"};
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session{env, policy_path.c_str(), session_options};

    auto input_names = read_io_names(session, true);
    auto output_names = read_io_names(session, false);
    auto input_name_ptrs = make_name_ptrs(input_names);
    auto output_name_ptrs = make_name_ptrs(output_names);

    std::vector<float> observation(kObservationSize, 0.0F);
    observation[0] = 1.0F;
    observation[24] = 0.5F;
    observation[33] = 1.0F;
    observation[36] = 0.25F;
    observation[40] = 0.15F;

    std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(kObservationSize)};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      observation.data(),
      observation.size(),
      input_shape.data(),
      input_shape.size());

    auto outputs = session.Run(
      Ort::RunOptions{nullptr},
      input_name_ptrs.data(),
      &input_tensor,
      1,
      output_name_ptrs.data(),
      1);

    if (outputs.empty() || !outputs.front().IsTensor()) {
      std::cerr << "Policy output is missing or not a tensor\n";
      return 1;
    }

    const auto shape_info = outputs.front().GetTensorTypeAndShapeInfo();
    const auto output_count = shape_info.GetElementCount();
    if (output_count != kActionSize) {
      std::cerr << "Expected " << kActionSize << " actions, got " << output_count << "\n";
      return 1;
    }

    const auto * output_data = outputs.front().GetTensorData<float>();
    for (std::size_t index = 0; index < output_count; ++index) {
      const float value = std::clamp(output_data[index], -1.0F, 1.0F);
      if (!std::isfinite(value)) {
        std::cerr << "Action " << index << " is not finite\n";
        return 1;
      }
      std::cout << (index == 0 ? "" : " ") << value;
    }
    std::cout << "\n";
  } catch (const Ort::Exception & error) {
    std::cerr << "ONNX Runtime error: " << error.what() << "\n";
    return 1;
  }

  return 0;
}
