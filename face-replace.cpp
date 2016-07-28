// Reads a video file and outputs its frames to a new file
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>

#include <string.h>
#include <iostream>

typedef std::vector<std::pair<cv::Rect, std::vector<cv::Rect>>> face_data_t;

// Removes all elements except the two largest regions
std::vector<cv::Rect> largest_regions(std::vector<cv::Rect> &eyes)
{
    std::vector<cv::Rect> largest_regions;
    if (eyes.size() < 2) {
        return largest_regions; // Empty vector
    } else if (eyes.size() > 2) {
        // Remove all regions except two largest.
        cv::Rect largest1 = eyes[0];
        cv::Rect largest2 = eyes[1];

        // We'll just use height as size factor
        if (largest2.height > largest1.height) {
            cv::Rect tmp = largest2;
            largest2 = largest1;
            largest1 = tmp;
        }

        for (int j = 2; j < eyes.size(); j++) {
            if (eyes[j].height > largest2.height) {
                if (eyes[j].height > largest1.height) {
                    cv::Rect tmp = largest1;
                    largest1 = eyes[j];
                    largest2 = tmp;
                } else {
                    largest2 = eyes[j];
                }
            }
        }
        largest_regions.push_back(largest2);
        largest_regions.push_back(largest1);
        return largest_regions;
    } else {
        return eyes;
    }
}

// Returns an array of pairs. Face paired with eyes.
#define CASCADE_PREFIX "/usr/share/opencv/haarcascades/"
#define FACE_CASCADE "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
#define EYE_CASCADE "/usr/share/opencv/haarcascades/haarcascade_eye.xml"
face_data_t face_detect(cv::Mat image) {
    // Convert image to gray
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, CV_BGR2GRAY);

    // Prepare face detectors
    cv::CascadeClassifier face_cascade(FACE_CASCADE);
    cv::CascadeClassifier eye_cascade(EYE_CASCADE);

    face_data_t face_data;
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray_image, faces, 1.1, 2);

    for (int i = 0; i < faces.size(); i++) {
        cv::Rect face_region = faces[i];
        int x = face_region.x;
        int y = face_region.y;
        int w = face_region.width;
        int h = face_region.height;

        // Generate a region of interest of face
        cv::Mat roi_gray(gray_image, face_region);

        // Detect eyes in face
        std::vector<cv::Rect> eyes;
        eye_cascade.detectMultiScale(roi_gray, eyes, 1.1, 2);

        if (eyes.size() == 0)
            continue;

        std::pair<cv::Rect, std::vector<cv::Rect>> face_datum(face_region, eyes);
        face_data.push_back(face_datum);
    }

    return face_data;
}

// http://answers.opencv.org/question/73016/how-to-overlay-an-png-image-with-alpha-channel-to-another-png/
void overlay_image(cv::Mat* src, cv::Mat* overlay, const cv::Point& location)
{
    // From location to image_edge/overlay_edge
    for (int y = cv::max(location.y, 0); y < src->rows; y++) {
        int fY = y - location.y;
        if (fY >= overlay->rows)
            break;

        for (int x = cv::max(location.x, 0); x < src->cols; x++) {
            int fX = x - location.x;
            if (fX >= overlay->cols)
                break;

            double opacity = ((double)overlay->data[fY * overlay->step + fX * overlay->channels() + 3]) / 255;

            for (int c = 0; opacity > 0 && c < src->channels(); c++) {
                unsigned char overlayPx = overlay->data[fY * overlay->step + fX * overlay->channels() + c];
                unsigned char srcPx = src->data[y * src->step + x * src->channels() + c];
                src->data[y * src->step + src->channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
            }
        }
    }
}

// Perform bounds check on overlay on image. If overlay is out of
// bounds, crop it. Modifies the given overlay
void crop_overlay(cv::Mat &image, cv::Mat &overlay, cv::Point location)
{
    cv::Size image_size = image.size();
    cv::Size overlay_size = overlay.size();
    cv::Rect cropped(0, 0, overlay_size.width, overlay_size.height);
    if (location.x < 0) {
        cropped.x += -location.x;
        cropped.width -= -location.x;
    }
    if (location.y < 0) {
        cropped.y += -location.y;
        cropped.height -= -location.y;
    }
    if (location.x + overlay_size.width > image_size.width) {
        cropped.width -= location.x + overlay_size.width - image_size.width;
    }
    if (location.y + overlay_size.height > image_size.height) {
        cropped.height -= location.y + overlay_size.height - image_size.height;
    }

    overlay = overlay(cropped);
}

#define SCALE 1.1
#define OFFSET 0.15
void face_replace(cv::Mat image, cv::Mat replacement, face_data_t &face_data)
{
    for (int i = 0; i < face_data.size(); i++) {
        cv::Rect face_region = face_data[i].first;
        int x = face_region.x;
        int y = face_region.y;
        int w = face_region.width;
        int h = face_region.height;

        // Resize replacement to have same height as image
        // Modify by SCALE
        cv::Size re_size = replacement.size();
        double aspect_ratio = (double)re_size.height / re_size.width;
        cv::Mat resized;
        cv::resize(replacement, resized, cv::Size(w * SCALE, w * aspect_ratio * SCALE));
        re_size = resized.size();

        // Copy to original image
        // Offset the replacement slightly to cover head.
        cv::Point location(x - re_size.width * OFFSET, y - re_size.height * OFFSET);
        if (replacement.channels() == 4) {
            overlay_image(&image, &resized, location);
        } else {
            crop_overlay(image, resized, location);
            re_size = resized.size(); // cropping has changed size.
            cv::Rect replace_region(cv::max(location.x, 0),
                                    cv::max(location.y, 0),
                                    re_size.width, re_size.height);
            resized.copyTo(image(replace_region));
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("face-replace IMAGE FACE_DATA");
        return 1;
    }

    // Read image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    cv::Mat replacement = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
    if (!image.data) {
        printf("Error loading image %s\n", argv[1]);
        return 2;
    }
    if (!replacement.data) {
        printf("Error loading image %s\n", argv[2]);
        return 2;
    }

    face_data_t face_data = face_detect(image);
    face_replace(image, replacement, face_data);

    cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
    cv::imshow("image", image);
    cv::waitKey(0);
    return 0;
}
