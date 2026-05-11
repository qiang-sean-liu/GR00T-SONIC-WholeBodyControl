#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "../include/control_policy.hpp"
#include "../include/encoder.hpp"

namespace {

constexpr std::uint32_t kReadyMagic = 0x31545254;  // "TRT1" little-endian
constexpr std::uint32_t kProtocolVersion = 1;

bool ReadExact(int fd, void* data, std::size_t bytes) {
  auto* ptr = static_cast<unsigned char*>(data);
  while (bytes > 0) {
    const ssize_t n = ::read(fd, ptr, bytes);
    if (n == 0) {
      return false;
    }
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    ptr += n;
    bytes -= static_cast<std::size_t>(n);
  }
  return true;
}

bool WriteExact(int fd, const void* data, std::size_t bytes) {
  const auto* ptr = static_cast<const unsigned char*>(data);
  while (bytes > 0) {
    const ssize_t n = ::write(fd, ptr, bytes);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    ptr += n;
    bytes -= static_cast<std::size_t>(n);
  }
  return true;
}

bool WriteU32(int fd, std::uint32_t value) {
  return WriteExact(fd, &value, sizeof(value));
}

bool ReadU32(int fd, std::uint32_t& value) {
  return ReadExact(fd, &value, sizeof(value));
}

void Usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --encoder <model_encoder.onnx> --decoder <model_decoder.onnx>"
      << " --response_fd <fd> [--encoder_fp16] [--decoder_fp16]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string encoder_path;
  std::string decoder_path;
  int response_fd = -1;
  bool encoder_fp16 = false;
  bool decoder_fp16 = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--encoder" && i + 1 < argc) {
      encoder_path = argv[++i];
    } else if (arg == "--decoder" && i + 1 < argc) {
      decoder_path = argv[++i];
    } else if (arg == "--response_fd" && i + 1 < argc) {
      response_fd = std::stoi(argv[++i]);
    } else if (arg == "--encoder_fp16") {
      encoder_fp16 = true;
    } else if (arg == "--decoder_fp16") {
      decoder_fp16 = true;
    } else if (arg == "-h" || arg == "--help") {
      Usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n";
      Usage(argv[0]);
      return 2;
    }
  }

  if (encoder_path.empty() || decoder_path.empty() || response_fd < 0) {
    Usage(argv[0]);
    return 2;
  }

  EncoderEngine encoder;
  PolicyEngine decoder;
  if (!encoder.Initialize(encoder_path, encoder_fp16)) {
    std::cerr << "Failed to initialize TensorRT encoder.\n";
    return 1;
  }
  if (!decoder.Initialize(decoder_path, decoder_fp16)) {
    std::cerr << "Failed to initialize TensorRT decoder.\n";
    return 1;
  }

  const std::uint32_t enc_in_dim = static_cast<std::uint32_t>(encoder.GetInputDimension());
  const std::uint32_t token_dim = static_cast<std::uint32_t>(encoder.GetTokenDimension());
  const std::uint32_t dec_in_dim = static_cast<std::uint32_t>(decoder.GetInputDimension());
  const std::uint32_t action_dim = static_cast<std::uint32_t>(decoder.GetActionDimension());

  if (!WriteU32(response_fd, kReadyMagic) ||
      !WriteU32(response_fd, kProtocolVersion) ||
      !WriteU32(response_fd, enc_in_dim) ||
      !WriteU32(response_fd, token_dim) ||
      !WriteU32(response_fd, dec_in_dim) ||
      !WriteU32(response_fd, action_dim)) {
    std::cerr << "Failed to write bridge ready header.\n";
    return 1;
  }

  while (true) {
    unsigned char request_type = 0;
    if (!ReadExact(STDIN_FILENO, &request_type, sizeof(request_type))) {
      break;
    }
    if (request_type == 'Q') {
      break;
    }

    std::uint32_t input_count = 0;
    if (!ReadU32(STDIN_FILENO, input_count)) {
      std::cerr << "Failed to read request input count.\n";
      return 1;
    }

    std::vector<float> input(input_count);
    if (!ReadExact(STDIN_FILENO, input.data(), input.size() * sizeof(float))) {
      std::cerr << "Failed to read request input tensor.\n";
      return 1;
    }

    if (request_type == 'E') {
      if (input_count != enc_in_dim) {
        std::cerr << "Encoder input count mismatch: got " << input_count
                  << ", expected " << enc_in_dim << "\n";
        return 1;
      }
      auto& input_buffer = encoder.GetInputBuffer();
      std::copy(input.begin(), input.end(), input_buffer.begin());
      if (!encoder.Encode()) {
        std::cerr << "Encoder inference failed.\n";
        return 1;
      }
      const auto& output = encoder.GetTokenBuffer();
      const unsigned char response_type = 'E';
      if (!WriteExact(response_fd, &response_type, sizeof(response_type)) ||
          !WriteU32(response_fd, token_dim) ||
          !WriteExact(response_fd, output.data(), output.size() * sizeof(float))) {
        std::cerr << "Failed to write encoder response.\n";
        return 1;
      }
    } else if (request_type == 'D') {
      if (input_count != dec_in_dim) {
        std::cerr << "Decoder input count mismatch: got " << input_count
                  << ", expected " << dec_in_dim << "\n";
        return 1;
      }
      auto& input_buffer = decoder.GetInputBuffer();
      std::copy(input.begin(), input.end(), input_buffer.begin());
      if (!decoder.Infer()) {
        std::cerr << "Decoder inference failed.\n";
        return 1;
      }
      const auto& output = decoder.GetActionBuffer();
      const unsigned char response_type = 'D';
      if (!WriteExact(response_fd, &response_type, sizeof(response_type)) ||
          !WriteU32(response_fd, action_dim) ||
          !WriteExact(response_fd, output.data(), output.size() * sizeof(float))) {
        std::cerr << "Failed to write decoder response.\n";
        return 1;
      }
    } else {
      std::cerr << "Unknown request type: " << static_cast<int>(request_type) << "\n";
      return 1;
    }
  }

  return 0;
}
