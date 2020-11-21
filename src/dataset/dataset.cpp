#include "dataset.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>

#include <map>
#include <unordered_map>

#include <glob.h>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <filesystem>
#include <omp.h>

/*
    This file contains code I used to create a labeled dataset from a set of ASCII tiles and a directory of (preprocessed) images. 
    
    The metric used to determine the best-fitting tile for an input image-slice uses SSIM at its core.
    It calculates a weighted SSIM in multiple places within a larger window than the actual character tile size.
    The best character within the larger window "wins". This allows for tolerance of slight misalignment from character and actual shape.

    Not the most clean code, I'm aware!
*/

#define ERROR(msg) std::cerr << "Error: " <<  msg << std::endl;  return 1;

#define PAD_WIDTH 3
#define N_TILES 95

using namespace cv;

struct patch {
    Mat img;
    Scalar mean = 0, std_dev = 0; // Used for calculating SSIM.
    int ind = -1;
};

static std::vector<patch> tileset;
static uint8_t n_tiles, n_rows, n_cols, n_pixels; // ASCII tile properties (all tiles must be the same size)

// SSIM parameters. Can be played with.
static float l_exp = 10;
static float cs_exp = 0.4;


static float gaussian_kernel[7][7] = 
/*
    Weight factors used when calculating SSIM for an input slice that's larger than tile size.
    Specifically for a padding size 3px. (Slices aren't padded with default values, but their actual environment in the original image!)
*/
{
{0.011362,	0.014962,	0.017649,	0.018648,	0.017649,	0.014962,	0.011362},
{0.014962,	0.019703,	0.02324	,	0.024556,	0.02324	,	0.019703,	0.014962},
{0.017649,	0.02324,    0.027413,	0.028964,	0.027413,	0.02324	,	0.017649},
{0.018648,	0.024556,   0.028964,	0.030603,	0.028964,	0.024556,	0.018648},
{0.017649,	0.02324,    0.027413,	0.028964,	0.027413,	0.02324	,	0.017649},
{0.014962,	0.019703,	0.02324	,	0.024556,	0.02324	,	0.019703,	0.014962},
{0.011362,	0.014962,	0.017649,	0.018648,	0.017649,	0.014962,	0.011362}

/*
// This one is found to produce slightly worse results.

{0.014786,	0.017272,	0.018961,	0.019559,	0.018961,	0.017272,	0.014786},
{0.017272,	0.020177,	0.022149,	0.022849,	0.022149,	0.020177,	0.017272},
{0.018961,	0.022149,	0.024314,	0.025082,	0.024314,	0.022149,	0.018961},
{0.019559,	0.022849,	0.025082,	0.025874,	0.025082,	0.022849,	0.019559},
{0.018961,	0.022149,	0.024314,	0.025082,	0.024314,	0.022149,	0.018961},
{0.017272,	0.020177,	0.022149,	0.022849,	0.022149,	0.020177,	0.017272},
{0.014786,	0.017272,	0.018961,	0.019559,	0.018961,	0.017272,	0.014786}
*/
};


int load_tileset_file(const std::string& fp) {
    
    FILE* file = fopen(fp.c_str(), "rb");

    fread(&n_tiles,sizeof(uint8_t), 1, file);
    fread(&n_rows, sizeof(uint8_t), 1, file);
    fread(&n_cols, sizeof(uint8_t), 1, file);

    n_pixels = n_rows * n_cols;

    for (int i = 0; i < n_tiles; ++i) {
        
        patch p;
        p.ind = i;

        uint8_t* arr = (uint8_t*) malloc(n_pixels * sizeof(uint8_t));

        fread(arr, sizeof(uint8_t), n_pixels, file);

        p.img = Mat(n_rows, n_cols, CV_8U, arr);

        meanStdDev(p.img, p.mean, p.std_dev);

        tileset.push_back(p);
    }

    return 0;
}

int load_tileset_dir(const std::string& dir, const std::string& ext) {

    std::string pattern = dir + "/*" + ext;
    glob_t paths = {0};
    
    if (glob(pattern.c_str(), GLOB_NOSORT, nullptr, &paths) != 0) {
        ERROR("Glob error or no tiles found");
    }

    n_tiles = paths.gl_pathc;

    for (int i = 0; i < n_tiles; ++i) {
        
        patch p;
        p.ind = i;

        p.img = imread(paths.gl_pathv[i], IMREAD_GRAYSCALE);

        if (i == 0) {
            n_rows = p.img.rows;
            n_cols = p.img.cols;
            n_pixels = n_rows * n_cols;
        }

        meanStdDev(p.img, p.mean, p.std_dev);

        tileset.push_back(p);
    }

    globfree(&paths);

    return 0;
}

int save_tileset(const std::string& fp) {

    if (n_rows == 0 || n_cols == 0) {
        ERROR("No tileset found");
    }

    std::ofstream of(fp);

    of.put(n_tiles);
    of.put(n_rows);;
    of.put(n_cols);


    for (int i = 0; i < n_tiles; ++i) {
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                of.put(tileset[i].img.at<uchar>(r, c));
            }
        }
    }

    of.close();

    return 0;
}

double ssim(patch p1, patch p2) {

    double best = -1;

    double L, CS, cov, res;
    patch p1_tmp;


    for (int c = 0; c < 2 * PAD_WIDTH + 1; ++c) {
        for (int r = 0; r < 2 * PAD_WIDTH + 1; ++r) {

            p1_tmp = cv::Mat(p1.img, cv::Rect(c, r, n_cols, n_rows));

            cv::meanStdDev(p1_tmp.img, p1_tmp.mean, p1_tmp.std_dev);

            if (p1_tmp.mean[0] == 0 && p2.mean[0] == 0) {
                L = 1;
            }
            else {
                L = 2 * p1_tmp.mean[0] * p2.mean[0] / ( p1_tmp.mean[0] * p1_tmp.mean[0] + p2.mean[0] * p2.mean[0] );
            }

            cov = 0;

            for (int i = 0; i < n_rows; ++i) {
                for (int j = 0; j < n_cols; ++j) {
                    cov += (p1_tmp.img.at<uchar>(i, j) - p1_tmp.mean[0]) * (p2.img.at<uchar>(i, j) - p2.mean[0]);
                }
            }

            cov /= n_rows * n_cols - 1;

            if (p1_tmp.std_dev[0] == 0 && p2.std_dev[0] == 0) {
                CS = 1;
            }
            else {
                    CS = 2 * cov / (p1_tmp.std_dev[0] * p1_tmp.std_dev[0] + p2.std_dev[0] * p2.std_dev[0]);
                }


            double res = ( ( pow(L, l_exp) * pow(CS, cs_exp) ) )*gaussian_kernel[r][c];
    
            #pragma omp critical
                {
                    if (res > best) {
                        best = res;
                    }
                }

        }
    }


    for (int i = 0; i < 49; ++i) {
        c = i / 7;
        r = i % 7;

    }    

    return best; 
}

std::vector<Mat> load_images(const std::string& dir, const std::string& ext) {
   
    std::string pattern = dir + "/*" + ext;
    glob_t paths = {0};

    std::vector<Mat> imgs;
    
    if (glob(pattern.c_str(), GLOB_NOSORT, nullptr, &paths) != 0) {
        std::cerr << "Warning: Glob error or no images found" << std::endl;
        exit(1);
    }

    for (int i = 0; i < paths.gl_pathc; ++i) {
        Mat img = imread(paths.gl_pathv[i], IMREAD_GRAYSCALE);
        imgs.push_back(img);
    }

    globfree(&paths);

    return imgs;
}

std::vector<patch> get_slices(const Mat& img, uint8_t n_rows, uint8_t n_cols, uint8_t pad_rows = 0, uint8_t pad_cols = 0, bool only_full_pad = false) {
    
    std::vector<patch> slices;

    int h_stride = n_cols + 2*pad_cols;
    int v_stride = n_rows + 2*pad_rows;


    for (int y = 0; y <= img.rows - v_stride; y += n_rows) {


        for (int x = 0; x <= img.cols - h_stride; x += n_cols) {

            Rect rect(x, y, h_stride, v_stride);

            patch p;

            p.img = Mat(img, rect);

            cv::meanStdDev(p.img, p.mean, p.std_dev);

            slices.push_back(p);
        }
    }

    return slices;
}

std::vector<patch> get_slices(const std::vector<Mat>& imgs, uint8_t n_rows, uint8_t n_cols, uint8_t pad_rows = 0, uint8_t pad_cols = 0, bool only_full_pad = false) {

    std::vector<patch> slices;

    for(auto img: imgs) {
        std::vector<patch> tmp = get_slices(img, n_rows, n_cols, pad_rows, pad_cols, only_full_pad);
        slices.insert(slices.end(), tmp.begin(), tmp.end());
    }

    return slices;
}

patch get_closest_tile(patch p) {

    if (tileset.empty()) {
        std::cerr << "Error: Tileset isn't loaded" << std::endl;
        exit(1);
    }


    double best_ssim = -1;
    patch closest_tile;

    #pragma omp parallel for 
    for (int i = 0; i < N_TILES; ++i) {

        double res = ssim(p, tileset[i]);

        #pragma omp critical 
        {
            if (res > best_ssim) {
                best_ssim = res;
                closest_tile = tileset[i];
            }
        }

    }



    return closest_tile;
}

cv::Mat display_closest_ascii_img(const Mat& img, bool pad = false) {

    Mat ascii_img = Mat(img.rows, img.cols, CV_8U);


    uint8_t pad_rows, pad_cols;
    
    if (pad) {
        pad_cols = pad_rows = 3;
    }
    else {
        pad_cols = pad_rows = 0;
    }


    std::vector<patch> slices = get_slices(img, n_rows, n_cols, pad_rows, pad_cols);

    std::vector<patch>::iterator it = slices.begin();



    for (int y = 0; y < (img.rows - 2*pad_rows) / n_rows; ++y) {
            for (int x = 0; x < (img.cols - 2*pad_cols) / n_cols; ++x) {
                Rect rect(x*n_cols, y*n_rows, n_cols, n_rows);


                
                patch closest_tile = get_closest_tile(*it);
                it++;

                Mat slice = ascii_img(rect);

                closest_tile.img.copyTo(slice);
            }
    }

    
    imshow("asciified image", ascii_img);
    waitKey(0);

    return ascii_img;
}

void create_labeled_dataset(const std::string& input_dir, const std::string& ext, const std::string& output_dir, bool pad = true, int samples_per_class = 10000, bool invert = true) {

    if (tileset.empty()) {
        std::cerr << "Error: No tileset loaded" << std::endl;
        exit(1);
    }

    std::string pattern = input_dir + "/*" + ext;
    glob_t paths = {0};

    Mat img;
    std::vector<Mat> slices;

    std::unordered_map<int, int> xy_pairs;
    std::map<int, int> class_cnt;
    
    if (glob(pattern.c_str(), GLOB_NOSORT, nullptr, &paths) != 0) {
        std::cerr << "Warning: Glob error or no images found" << std::endl;
        exit(1);
    }

    std::filesystem::create_directory(output_dir);

    int classes_left = n_tiles;

    uint8_t pad_cols, pad_rows;

    if (pad) {
        pad_cols = pad_rows = 3;
    }
    else {
        pad_cols = pad_rows = 0;
    }
 
    for (int i = 0; i < paths.gl_pathc; ++i) {

        std::cout << "Classes left after " << i << " files: " << classes_left << std::endl;

        if (!classes_left) break;

        Mat img = imread(paths.gl_pathv[i], IMREAD_GRAYSCALE);

        if (invert) img = ~img;
        
        std::vector<patch> curr_slices = get_slices(img, n_rows, n_cols, pad_cols, pad_rows, true);

        for (patch p: curr_slices) {

            if (p.mean[0] == 255) {
            //   continue;
            }

            patch closest_tile = get_closest_tile(p);

            int ind = closest_tile.ind;

            if (class_cnt[ind] < samples_per_class) {
                slices.push_back(p.img);
                
                xy_pairs[slices.size() - 1] = ind;
                class_cnt[ind] += 1;

                if (class_cnt[ind] == samples_per_class) {
                    classes_left--;
                }
            }
        }
     
    }

    globfree(&paths);

    save_tileset(output_dir + "/tileset.dat");

    std::ofstream of(output_dir + "/xy_pairs.dat");

    int slice_rows = slices[0].rows;
    int slice_cols = slices[0].cols;

    of.write(reinterpret_cast<char*>(&slice_rows), 4);
    of.write(reinterpret_cast<char*>(&slice_cols), 4);

    for (auto xy: xy_pairs) {
        
        int slice_ind = xy.first;
        int class_ind = xy.second;

        Mat norm_slice;
        
        slices[slice_ind].convertTo(norm_slice, CV_32F, 1.f/255);

        for (int r = 0; r < slice_rows; ++r) {
            for (int c = 0; c < slice_cols; ++c) {
                of.write(reinterpret_cast<char*>(&norm_slice.at<float>(r, c)), 4);
            }
        }

        of.write(reinterpret_cast<char*>(&class_ind), 4);

    }    

    return;
}

