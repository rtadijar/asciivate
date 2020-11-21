#include <string>
#include <vector>

#include <opencv4/opencv2/core.hpp>

struct patch;

int load_tileset_file(const std::string& fp);

int load_tileset_dir(const std::string& dir, const std::string& ext);

int save_tileset(const std::string& fp);

std::vector<cv::Mat> load_images(const std::string& dir, const std::string& ext);

std::vector<patch> get_slices(const cv::Mat& img, uint8_t n_rows, uint8_t n_cols, uint8_t pad_rows, uint8_t pad_cols, bool only_full_pad);

std::vector<patch> get_slices(const std::vector<cv::Mat>& imgs, uint8_t n_rows, uint8_t n_cols, uint8_t pad_rows, uint8_t pad_cols, bool only_full_pad);

double ssim(patch p1, patch p2);

patch get_closest_tile(patch p);

cv::Mat display_closest_ascii_img(const cv::Mat& img, bool pad);

void create_labeled_dataset(const std::string& input_dir, const std::string& ext, const std::string& output_dir, bool pad, int samples_per_class);

