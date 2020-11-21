#include <iostream>
#include <fstream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudafilters.hpp>

/*
    This program demonstrates real-time conversion to structural ASCII art. Calling without arguments converts the default camera stream into ASCII and displays it back to the user.

    Note: The manner in which OpenCV reads camera frames and cv::imshow() represent a potential bottleneck in display (i.e. the apparent framerate may be lower than what is possible with the available resources).

    Additionally, the program can be called in this manner:
        $ asciivate src dst
    where src and dst represent paths to the input video and the converted video, respectively.

    Note: dst has to have the .avi extension and OpenCV must have access to the MJPG codec!
*/


extern void convert();
extern void initialize_converter(float* _h_src, uchar* _h_dst, int rows, int cols, const std::string& classifier_path = "model.dat", const std::string& texture_path = "texture.png");
extern void free_converter();

// Parameters for the edge detection filter.
int canny_thresh_low = 15;
int canny_thresh_high = 30;

std::string ascii_window_name = "Asciivate";
std::string settings_window_name = "Settings";

cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edge_detector = cv::cuda::createCannyEdgeDetector(canny_thresh_low, canny_thresh_high); 
cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(CV_8U, CV_8U, cv::Size(5, 5), 0);

// GUI setup

static void on_canny_thresh_low_change(int, void*) {

    canny_thresh_low = std::min(canny_thresh_low, canny_thresh_high - 1);
    canny_edge_detector->setLowThreshold(canny_thresh_low);

    cv::setTrackbarPos("canny_low", settings_window_name, canny_thresh_low);
}

static void on_canny_thresh_high_change(int, void*) {

    canny_thresh_high = std::max(canny_thresh_high, canny_thresh_low + 1);
    canny_edge_detector->setHighThreshold(canny_thresh_high);

    cv::setTrackbarPos("canny_high", settings_window_name, canny_thresh_high);
}

void initialize_windows() {
    cv::namedWindow(ascii_window_name);
    cv::namedWindow(settings_window_name, cv::WINDOW_GUI_NORMAL);

    cv::resizeWindow(settings_window_name, cv::Size(500, 70));

    cv::createTrackbar("canny_low", settings_window_name, &canny_thresh_low, 300, on_canny_thresh_low_change);
    cv::createTrackbar("canny_high", settings_window_name, &canny_thresh_high, 300, on_canny_thresh_high_change);
}


/*
    The preprocessing that follows mimics the preprocessing used when the training set for the neural classifier was generated.

    The model won't perform well on regular grayscale images!
*/

void preprocess_img_cuda(cv::Mat src, cv::Mat dst) { 
    cv::cuda::GpuMat tmp;  
    tmp.upload(src);

    cv::cuda::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY); // We work with grayscale.
    cv::cuda::bilateralFilter(tmp, tmp, 7, 300, 300); // Smooths out noise while preserving edges.

    canny_edge_detector->detect(tmp, tmp); // Extracts edges.
    gaussian_filter->apply(tmp, tmp); // Just a simple blur.
    
    tmp.download(src);

    src = ~src;
    src.convertTo(dst, CV_32F, 1./255);
}

void preprocess_img(cv::Mat src, cv::Mat dst, int canny_low, int canny_high) {   
    /*
        Same as above, but not using the CUDA OpenCV API. Comparatively slow, not fit for real-time.
    */

    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY); 
    cv::Mat tmp;
    
    
    cv::bilateralFilter(src, tmp, 7, 300, 300); 
    cv::Canny(tmp, src, canny_thresh_low, canny_thresh_high); 

    cv::GaussianBlur(src, src, cv::Size(3,3), 0); 
    src = ~src;
    src.convertTo(dst, CV_32F, 1./255);
}


int main(int argc, const char* argv[]) {

    cv::VideoCapture cap;
    cv::VideoWriter writer;

    cv::Mat frame, processed, ascii;

    int rows, cols;
    int fps = 25;

    bool writing = false;

    if (argc == 1) { // Converting camera stream.
        cap.open(0);

        if (!cap.isOpened()) {
            std::cerr << "error: couldn't open camera stream" << std::endl;
            return 1;
        }
    }
    else if (argc == 3) { // Converting a video file.
        cap.open(argv[1]);
        
        if (!cap.isOpened()) {
            std::cerr << "error: couldn't open video stream" << std::endl;
            return 1;
        }

        fps = cap.get(cv::CAP_PROP_FPS);
        rows = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        cols = cap.get(cv::CAP_PROP_FRAME_WIDTH);

        writer.open(argv[2], cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(cols, rows), false);

        if (!writer.isOpened()) {
            std::cerr << "error: couldn't open output stream (note: only '.avi' extension is supported)" << std::endl;
            return 1;
        }

        writing = true;
    }


    if (cap.read(frame) == false) {
        cap.release();
        return 1;
    }
    else {
        cols = frame.cols;
        rows = frame.rows;

        processed = cv::Mat(rows, cols, CV_32F);
        ascii = cv::Mat(rows, cols, CV_8U);
        
        initialize_converter(processed.ptr<float>(0), ascii.ptr<uchar>(0), rows, cols); // Bind the converter to the appropriate input/output.
        initialize_windows();
    }

    while(true) { 
        if (cap.read(frame) == false) {
            cap.release();
            break;
        }

        preprocess_img_cuda(frame, processed); // Extract edges and deliver to converter.
        convert();  // The magic stuff.

        cv::imshow(ascii_window_name, ascii);
        if (writing) writer.write(ascii);

        int k = cv::waitKey(1000/fps) & 0xff;

        if (k == 27) break;
    }

    if (cap.isOpened()) cap.release();
    if (writer.isOpened()) writer.release();

    cv::destroyAllWindows();

    free_converter();
}



