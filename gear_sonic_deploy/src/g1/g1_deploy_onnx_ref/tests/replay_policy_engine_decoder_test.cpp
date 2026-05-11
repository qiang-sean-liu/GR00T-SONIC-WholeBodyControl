#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "../include/control_policy.hpp"
#include "../include/policy_parameters.hpp"

namespace {

constexpr char kMagic[8] = {'S', 'O', 'N', 'I', 'C', 'D', 'E', 'C'};
constexpr std::uint32_t kVersion = 1;

struct Header {
  char magic[8];
  std::uint32_t version;
  std::uint32_t reserved;
  std::uint64_t frames;
  std::uint64_t input_dim;
  std::uint64_t action_dim;
};

struct Stats {
  double sum_l2 = 0.0;
  double max_l2 = 0.0;
  double sum_max_abs = 0.0;
  double max_abs = 0.0;
  std::uint64_t count = 0;
};

void UpdateStats(Stats& stats, const std::vector<double>& diff) {
  double l2_sq = 0.0;
  double max_abs = 0.0;
  for (double v : diff) {
    l2_sq += v * v;
    max_abs = std::max(max_abs, std::abs(v));
  }
  const double l2 = std::sqrt(l2_sq);
  stats.sum_l2 += l2;
  stats.max_l2 = std::max(stats.max_l2, l2);
  stats.sum_max_abs += max_abs;
  stats.max_abs = std::max(stats.max_abs, max_abs);
  stats.count += 1;
}

template <typename T>
bool ReadVector(std::ifstream& in, std::vector<T>& out, std::uint64_t count) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
    return false;
  }
  out.resize(static_cast<std::size_t>(count));
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(T)));
  return static_cast<bool>(in);
}

void Usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --model <model_decoder.onnx> --input <decoder_test.bin> [--fp16]\n"
      << "\n"
      << "Runs recorded decoder_obs through the deploy PolicyEngine TensorRT path and\n"
      << "compares against recorded sonic.decoder_action_raw and sonic.q_target_cmd.\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string input_path;
  bool fp16 = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--fp16") {
      fp16 = true;
    } else if (arg == "-h" || arg == "--help") {
      Usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n";
      Usage(argv[0]);
      return 2;
    }
  }

  if (model_path.empty() || input_path.empty()) {
    Usage(argv[0]);
    return 2;
  }

  std::ifstream in(input_path, std::ios::binary);
  if (!in) {
    std::cerr << "Failed to open input file: " << input_path << "\n";
    return 1;
  }

  Header header{};
  in.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!in) {
    std::cerr << "Failed to read test input header.\n";
    return 1;
  }
  if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0 || header.version != kVersion) {
    std::cerr << "Invalid test input format.\n";
    return 1;
  }
  if (header.input_dim != 994 || header.action_dim != G1_NUM_MOTOR) {
    std::cerr << "Unexpected dims: input_dim=" << header.input_dim
              << " action_dim=" << header.action_dim << "\n";
    return 1;
  }

  const std::uint64_t obs_count = header.frames * header.input_dim;
  const std::uint64_t action_count = header.frames * header.action_dim;

  std::vector<float> decoder_obs;
  std::vector<float> recorded_action;
  std::vector<double> recorded_q_target;
  if (!ReadVector(in, decoder_obs, obs_count) ||
      !ReadVector(in, recorded_action, action_count) ||
      !ReadVector(in, recorded_q_target, action_count)) {
    std::cerr << "Failed to read decoder test tensors.\n";
    return 1;
  }

  PolicyEngine policy_engine;
  if (!policy_engine.Initialize(model_path, fp16)) {
    std::cerr << "Failed to initialize PolicyEngine.\n";
    return 1;
  }
  if (policy_engine.GetInputDimension() != header.input_dim ||
      policy_engine.GetActionDimension() != header.action_dim) {
    std::cerr << "PolicyEngine dims do not match input file: input="
              << policy_engine.GetInputDimension() << " action="
              << policy_engine.GetActionDimension() << "\n";
    return 1;
  }

  Stats raw_stats;
  Stats q_stats;
  std::vector<double> first_raw_diff(header.action_dim, 0.0);
  std::vector<double> first_q_diff(header.action_dim, 0.0);
  std::vector<float> first_raw_output(header.action_dim, 0.0f);
  std::vector<double> first_q_output(header.action_dim, 0.0);

  auto& input_buffer = policy_engine.GetInputBuffer();
  auto& action_buffer = policy_engine.GetActionBuffer();

  for (std::uint64_t f = 0; f < header.frames; ++f) {
    const std::size_t obs_offset = static_cast<std::size_t>(f * header.input_dim);
    for (std::size_t j = 0; j < static_cast<std::size_t>(header.input_dim); ++j) {
      input_buffer[j] = decoder_obs[obs_offset + j];
    }

    if (!policy_engine.Infer()) {
      std::cerr << "PolicyEngine inference failed at frame " << f << "\n";
      return 1;
    }

    std::vector<double> raw_diff(header.action_dim, 0.0);
    std::vector<double> q_diff(header.action_dim, 0.0);
    std::array<double, G1_NUM_MOTOR> q_output{};

    const std::size_t action_offset = static_cast<std::size_t>(f * header.action_dim);
    for (std::size_t j = 0; j < static_cast<std::size_t>(header.action_dim); ++j) {
      raw_diff[j] = static_cast<double>(action_buffer[j]) -
                    static_cast<double>(recorded_action[action_offset + j]);
    }

    for (std::size_t mujoco_i = 0; mujoco_i < G1_NUM_MOTOR; ++mujoco_i) {
      const double action_value =
          static_cast<double>(action_buffer[isaaclab_to_mujoco[mujoco_i]]) *
          g1_action_scale[mujoco_i];
      q_output[mujoco_i] = static_cast<double>(static_cast<float>(
          default_angles[mujoco_i] +
          action_value));
      q_diff[mujoco_i] = q_output[mujoco_i] -
                         recorded_q_target[action_offset + mujoco_i];
    }

    UpdateStats(raw_stats, raw_diff);
    UpdateStats(q_stats, q_diff);

    if (f == 0) {
      first_raw_diff = raw_diff;
      first_q_diff = q_diff;
      for (std::size_t j = 0; j < static_cast<std::size_t>(header.action_dim); ++j) {
        first_raw_output[j] = action_buffer[j];
        first_q_output[j] = q_output[j];
      }
    }
  }

  auto print_stats = [](const std::string& name, const Stats& stats) {
    std::cout << name << ":\n"
              << "  frames=" << stats.count << "\n"
              << "  mean_l2=" << std::setprecision(12) << (stats.sum_l2 / stats.count) << "\n"
              << "  max_l2=" << std::setprecision(12) << stats.max_l2 << "\n"
              << "  mean_max_abs=" << std::setprecision(12) << (stats.sum_max_abs / stats.count) << "\n"
              << "  max_abs=" << std::setprecision(12) << stats.max_abs << "\n";
  };

  std::cout << "PolicyEngine decoder replay test\n"
            << "  model=" << model_path << "\n"
            << "  input=" << input_path << "\n"
            << "  precision=" << (fp16 ? "FP16" : "FP32") << "\n"
            << "  frames=" << header.frames << "\n";
  print_stats("raw_action_vs_recorded", raw_stats);
  print_stats("q_target_vs_recorded", q_stats);

  std::cout << "first_frame_raw_action_first8:";
  for (std::size_t i = 0; i < std::min<std::size_t>(8, first_raw_output.size()); ++i) {
    std::cout << ' ' << std::setprecision(9) << first_raw_output[i];
  }
  std::cout << "\nfirst_frame_raw_diff_first8:";
  for (std::size_t i = 0; i < std::min<std::size_t>(8, first_raw_diff.size()); ++i) {
    std::cout << ' ' << std::setprecision(9) << first_raw_diff[i];
  }
  std::cout << "\nfirst_frame_q_target_first8:";
  for (std::size_t i = 0; i < std::min<std::size_t>(8, first_q_output.size()); ++i) {
    std::cout << ' ' << std::setprecision(9) << first_q_output[i];
  }
  std::cout << "\nfirst_frame_q_diff_first8:";
  for (std::size_t i = 0; i < std::min<std::size_t>(8, first_q_diff.size()); ++i) {
    std::cout << ' ' << std::setprecision(9) << first_q_diff[i];
  }
  std::cout << "\n";

  return 0;
}
