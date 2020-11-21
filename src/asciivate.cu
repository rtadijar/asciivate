#include <cuda.h>
#include <fstream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>

/*
    This file contains the asciivating kernel and everything needed to get conversion running.
*/

#define LAYERS  4
#define ASCII_CHARS 95 // Printables

// Size of ascii character
static int tile_width = 7;
static int tile_height = 9;

// Size of input window for the classifier 
static int window_height = 15;
static int window_width = 13;

static int blank_ind = 80; // Index of blank tile in the tileset. Used when encountering completely white input as an optimization.

static int wt_diff = window_height - tile_height;

#define CUDA_MEASURE_START() cudaEvent_t start = cudaEvent_t();\
                             cudaEvent_t stop  = cudaEvent_t();\
                             check(cudaEventCreate( &start ));\
                             check(cudaEventCreate( &stop  ));\
                             check(cudaEventRecord( start, 0 ));

#define CUDA_MEASURE_END() check(cudaEventRecord( stop, 0));\
                           check(cudaEventSynchronize( stop ));\
                           float elapsed = 0.f;\
                           check(cudaEventElapsedTime( &elapsed, start, stop ));\
                           check(cudaEventDestroy(start));\
                           check(cudaEventDestroy(stop) ); std::cout << "Device execution time is " << elapsed << std::endl;
  
#define check(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


struct linear_model {
    // ReLU assumed for all other than output layer
    float* weights[LAYERS];
    float* biases[LAYERS];
    int size = LAYERS;
    int shapes[2*LAYERS]; // Stores the shapes of the linearized matrices
};

struct shape2d {
    int x;
    int y;
};

/*
    Converter state. 

    The converter binds to a host src/dst pointer where it reads/writes the processed/asciified image data. It handles the allocation of the corresponding device memory.

    Only one converter/model may be initialized at a time and it is not thread safe. This is not a problem for this application.
*/

static bool init = false; 

static int img_rows, img_cols;
static int block_size_x, block_size_y;

static linear_model classifier, device_classifier;
static cudaTextureObject_t tileset;

static float *d_src, *h_src;
static uchar *d_dst, *h_dst;

static size_t d_src_pitch, d_dst_pitch;


linear_model read_model_from_file(const std::string& fp) {
    /*
        Deserializes neural network parameters. Expects a binary file of floats in host machine endiannes.

        The memory layout of a serialized layer is: [n][m][weight matrix][bias vector]
        where n and m are single floats representing the dimensions of the layer.
    */

    std::ifstream i_stream(fp);
    linear_model model;

    int layer = 0;
    do {

        int n, m;

        i_stream.read(reinterpret_cast<char*>(&n), 4);

        if (i_stream.eof()) {
            break;
        }

        i_stream.read(reinterpret_cast<char*>(&m), 4);

        assert(n > 0 && m > 0);

        model.weights[layer] = static_cast<float*>(malloc(sizeof (float) * n * m));
        model.biases[layer] = static_cast<float*>(malloc(sizeof(float) * m));


        i_stream.read(reinterpret_cast<char*>(model.weights[layer]), sizeof (float) * n * m);
        i_stream.read(reinterpret_cast<char*>(model.biases[layer]), sizeof (float) * m);
 
        
        model.shapes[layer*2] = n; 
        model.shapes[layer*2 + 1] = m; 

        layer++;
    } while (true);  

    model.size = layer;


    return model;
}

__host__ linear_model send_model_to_device(linear_model model) {
    linear_model device_model;

    device_model.size = model.size;
    
    for (int i = 0; i < LAYERS; ++i) {
        device_model.shapes[2*i] = model.shapes[2*i];
        device_model.shapes[2*i+1] = model.shapes[2*i + 1];

        check(cudaMalloc(&(device_model.weights[i]), sizeof(float) * device_model.shapes[2*i] * device_model.shapes[2*i + 1] ));
        check(cudaMalloc(&(device_model.biases[i]), sizeof(float) * device_model.shapes[2*i + 1] ));

        check(cudaMemcpy(device_model.weights[i], model.weights[i], sizeof(float) * device_model.shapes[2*i] * device_model.shapes[2*i + 1] , cudaMemcpyHostToDevice ));
        check(cudaMemcpy(device_model.biases[i], model.biases[i], sizeof(float) * device_model.shapes[2*i + 1], cudaMemcpyHostToDevice ));
    }

    return device_model;
}

void free_model_host(linear_model model) {
    for (int layer = 0; layer < LAYERS; ++layer) {
        free(classifier.weights[layer]);
        free(classifier.biases[layer]);
    }
}

__host__ void free_model_device(linear_model model) {
    for (int layer = 0; layer < LAYERS; ++layer) {
        check(cudaFree(device_classifier.weights[layer]));
        check(cudaFree(device_classifier.biases[layer]));
    }
}

__host__ cudaTextureObject_t send_texture_to_device(const std::string& fp) {
    /*
        Sets up the texture on the device.
    */

    cv::Mat tex_host = cv::imread(fp, cv::IMREAD_GRAYSCALE);

    uint8_t *tex_dev;
    size_t tex_pitch;

    check(cudaMallocPitch(&tex_dev, &tex_pitch, sizeof(uint8_t) * tex_host.cols, tex_host.rows));
    check(cudaMemcpy2D(tex_dev, tex_pitch, tex_host.ptr(0), sizeof(uchar) * tex_host.cols, sizeof(uchar) * tex_host.cols, tex_host.rows, cudaMemcpyHostToDevice));

    cudaResourceDesc res_desc;

    memset(&res_desc, 0, sizeof(res_desc));

    res_desc.resType = cudaResourceTypePitch2D;

    res_desc.res.pitch2D.devPtr = tex_dev;
    res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar>();
    res_desc.res.pitch2D.width = tex_host.cols;
    res_desc.res.pitch2D.height = tex_host.rows;
    res_desc.res.pitch2D.pitchInBytes = tex_pitch;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));

    cudaTextureObject_t tileset = 0;
    check(cudaCreateTextureObject(&tileset, &res_desc, &tex_desc, NULL));

    return tileset;
}

__global__ void asciivate_kernel(linear_model classifier, cudaTextureObject_t tileset, float* img, size_t img_pitch, uchar* output, size_t output_pitch, shape2d tile_shape, shape2d window_shape, int blank_ind) {
    /*
        A block is assigned a region of interest in the source image, and the corresponding output tile in the destination image.

        Blocks are completely independent.
    */

    __shared__ float arr[256];
    __shared__ int max_ind;
    __shared__ float luminance;

    int id = threadIdx.x;

    // Get input patch
    int window_sz = window_shape.x * window_shape.y;

    int x = id % window_shape.x;
    int y = id / window_shape.x;

    if(id == 0) {
        luminance = 0.; // Stores the total luminence of the input. 
        max_ind = blank_ind;
    }

    __syncthreads();
    
    if (id < window_sz) {
        arr[id] =  *( (float*)((char*) img + img_pitch * (blockIdx.y * tile_shape.y + y)) + blockIdx.x * tile_shape.x + x);
        atomicAdd_block(&luminance, arr[id]);
    }

    __syncthreads();

    /*
        As an optimization for most expected source images, an input window which is completely blank (i.e. luminance is max) is not classified.
        Instead, it is immediately replaced by the blank tile. 
    */
    if (luminance < window_sz) {

        int m, n;

        // Iterate over network layers. ReLU is assumed for all except output layer.
        for (int layer = 0; layer < LAYERS; ++layer) {
            __syncthreads();

            n = classifier.shapes[2*layer];
            m = classifier.shapes[2*layer + 1];

            float sum = 0;

            if (id < m) {
                for (int i = 0; i < n; ++i) {
                    sum += arr[i]*classifier.weights[layer][i * m + id];
                }
            }

            sum += classifier.biases[layer][id];

            __syncthreads();

            if (id < m) {
                if ( layer != LAYERS - 1 && sum < 0) sum = 0;

                arr[id] = sum;
            }
        }

        __syncthreads();


        int max = -10000;

        // Find the argmax of the output layer. Wastes blocks/threads but executes very quickly compared to classification.
        if (id == 0) {        
            for (int i = 0; i < m; ++i) {
                if (arr[i] > max) {
                    max = arr[i];
                    max_ind = i;
                }
            }
        }

        __syncthreads();
    }
    
    // Write best-fitting tile to output patch.
    int tile_sz = tile_shape.x * tile_shape.y;

    x = id % tile_shape.x;
    y = id / tile_shape.x;

    if (id < tile_sz) {
        *(uchar*)(output + output_pitch * (blockIdx.y * tile_shape.y + y) + blockIdx.x * tile_shape.x + x) = tex2D<uchar>(tileset, max_ind * tile_shape.x + x, y);
    }
}


void initialize_converter(float* _h_src, uchar* _h_dst, int rows, int cols, const std::string& classifier_path = "model.dat", const std::string& tileset_path = "texture.png") {
    if (init) return;

    classifier = read_model_from_file(classifier_path);
    device_classifier = send_model_to_device(classifier);
    tileset = send_texture_to_device(tileset_path);

    img_rows = rows;
    img_cols = cols;

    h_src = _h_src;
    h_dst = _h_dst;

    check(cudaMallocPitch(&d_src, &d_src_pitch, sizeof(float) * img_cols, img_rows));
    check(cudaMallocPitch(&d_dst, &d_dst_pitch, sizeof(uchar) * img_cols, img_rows));

    block_size_y = (img_rows - wt_diff) / tile_height;
    block_size_x = (img_cols - wt_diff) / tile_width;

    init = true;
}

void free_converter() {
    if (!init) return;

    free_model_host(classifier);
    free_model_device(device_classifier);

    check(cudaFree(d_src));
    check(cudaFree(d_dst));

    check(cudaDestroyTextureObject(tileset));

    init = false;
}

void convert() {
    /*
        Converts the image received in h_src and places the result in h_dst.
    */

    if (!init) return;

    check(cudaMemcpy2D(d_src, d_src_pitch, h_src, sizeof(float) * img_cols, sizeof(float) * img_cols,  img_rows, cudaMemcpyHostToDevice));

    dim3 gridDim(  block_size_x , block_size_y );
    dim3 blockDim( 256 ); // Must be at least the size of the largest network layer dimension.

    asciivate_kernel<<< gridDim, blockDim >>>(device_classifier, tileset, d_src, d_src_pitch, d_dst, d_dst_pitch, {tile_width, tile_height}, {window_width, window_height}, blank_ind);

    check(cudaMemcpy2D( h_dst, sizeof(uchar) * img_cols, d_dst, d_dst_pitch, sizeof(uchar) * img_cols, img_rows, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
}

