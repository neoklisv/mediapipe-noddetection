// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"

#include <deque>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <random>

#define NODDING_SENSITIVITY 0.0125
#define SHAKING_SENSITIVITY 0.018
#define VERTICAL_ADJUSTMENT 0.2
#define HORIZONTAL_ADJUSTMENT 0.12

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "multi_face_landmarks";
constexpr char kWindowName[] = "MediaPipe";

// function for detecting change in point direction
int scanDirection(const std::deque<mediapipe::NormalizedLandmark> &data, const double &sensitivity, bool nod)
{
  double curPoint;
  double prevPoint = -1;
  int curDirection;
  int prevDirection = -1;

  double inflection;

  inflection = data[0].z();

  int changes = 0;

  for (const auto &cur : data)
  {

    curPoint = cur.z();

    if (prevPoint != -1)
    {
      // If the two neighboring data points are significantly far away
      if (abs(inflection - curPoint) > sensitivity)
      {

        // check direction of travel
        if (inflection > curPoint)
          curDirection = 1;
        else
          curDirection = 0;

        // check if there has been a bidirecitonal change for shake and only down for nod
        if (!nod && prevDirection > -1 && curDirection != prevDirection ||
            nod && prevDirection > -1 && curDirection == 1 && prevDirection == 0)
        {

          // Assign a new inflection point as the current data point.
          changes += 1;
          inflection = curPoint;
        }

        else if (prevDirection == -1)
        {
          prevDirection = curDirection;
          inflection = curPoint;
        }
      }
    }
    prevPoint = curPoint;
  }

  return changes;
}

absl::Status RunMPPGraph()
{

  // load questions from text file
  std::vector<std::string> questions;
  std::string filename;
  {
    std::ifstream file("/facemesh_files/questions.txt");
    if (!file.is_open())
    {
      std::cerr << "Could not open the questions file" << std::endl;
    }
    std::string line;
    while (getline(file, line))
    {
      questions.push_back(line);
    }
    file.close();
  }

  // create new file for logging answers with current date/time
  {
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&in_time_t);
    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y-%m-%d_%H-%M-%S");
    std::string str_time = oss.str();
    filename = "/facemesh_files/logs/ex1_answers_" + str_time + ".csv";

    std::ofstream file(filename);
    if (file.is_open())
    {
      file << "question,answer,timestamp" << std::endl;
      file.close();
    }
    else
    {
      std::cerr << "Failed to create the file." << std::endl;
    }
  }

  // variable for dynamic frame quantity adjustment (to adjust for slower/faster systems)
  int framesToAnalyze = 8;

  // variables for storing landmarks for analysis
  std::deque<mediapipe::NormalizedLandmark> nodding_coordinates;
  std::deque<mediapipe::NormalizedLandmark> shaking_coordinates;

  // setup of mediapipe pipeline
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents("/facemesh_files/face_mesh_desktop_live.pbtxt",
      &calculator_graph_config_contents));
  ABSL_LOG(INFO) << "Get calculator graph config contents: "
                 << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  // load OpenCV camera stream
  ABSL_LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;


  capture.open(0);

  RET_CHECK(capture.isOpened());

  cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  capture.set(cv::CAP_PROP_FPS, 30);

  ABSL_LOG(INFO) << "Start running the calculator graph.";

  // setup polling from graphs for exctracting landmark coordinates as well as anotated image
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark,
                      graph.AddOutputStreamPoller(kLandmarksStream));

  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                      graph.AddOutputStreamPoller(kOutputStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;

  // start questions from random position
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, questions.size());

  // variables for question prompting
  int q_i = distrib(gen);
  int q_max = questions.size();
  std::string text = questions[q_i];
  std::string text2 = "";
  auto wait_interval = std::chrono::system_clock::now();
  auto prev_time = std::chrono::system_clock::now();
  bool ready_next = true;
  int time_index = 0;

  // main loop for each capured camera frame
  while (grab_frames)
  {
    // variable for logging user answer
    int answer = 0;

    // variable for dynamically adjusting frames to process
    time_index += 1;

    // Capture opencv camera frame and flip before processing
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;

    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    // landmark polling
    mediapipe::Packet landmark_packet;
    if (poller_landmark.QueueSize() > 0 && poller_landmark.Next(&landmark_packet))
    {

      // analyze available faces (setup for maximum one face currently via graph parameters)
      auto &output_landmark_faces = landmark_packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
      for (auto &face : output_landmark_faces)
      {

        // capturing points of interest for tracking head movements
        auto &chin = face.landmark(199);
        auto &sidehead = face.landmark(447);

        // for adjusting sensitivity constants based on distance to screen
        auto &tophead = face.landmark(10);
        auto &bottomhead = face.landmark(152);
        auto distance_adjustment = (bottomhead.y() - tophead.y()) / 0.5;

        // Always append a new frame to both lists.
        nodding_coordinates.push_back(chin);
        shaking_coordinates.push_back(sidehead);

        if (nodding_coordinates.size() > framesToAnalyze && shaking_coordinates.size() > framesToAnalyze)
        {
          // Pop the oldest frame from both lists (just looking at the last FRAMES_TO_ANALYZE frames).
          nodding_coordinates.pop_front();
          shaking_coordinates.pop_front();

          // find min/ax of nodding points
          double max_nod = 0;
          double min_nod = 99;
          for (auto &elem : nodding_coordinates)
          {
            if (elem.y() > max_nod)
              max_nod = elem.y();
            if (elem.y() < min_nod)
              min_nod = elem.y();
          }

          // find min/ax of shaking points
          double max_shake = 0;
          double min_shake = 99;
          for (auto &elem : shaking_coordinates)
          {
            if (elem.x() > max_shake)
              max_shake = elem.x();
            if (elem.x() < min_shake)
              min_shake = elem.x();
          }

          // save direction scans as variables to save time from multiple calls
          int nodScan = scanDirection(nodding_coordinates, SHAKING_SENSITIVITY * distance_adjustment, false);
          int shakeScan = scanDirection(shaking_coordinates, SHAKING_SENSITIVITY * distance_adjustment, false);

          // classify as nod only when having significant downwards vertical movement and direction change and no shaking movements
          if (scanDirection(nodding_coordinates, NODDING_SENSITIVITY * distance_adjustment, true) > 0 &&
              shakeScan == 0 &&
              abs(max_nod - min_nod) <= VERTICAL_ADJUSTMENT * distance_adjustment)
          {
            std::cout << "YES\n";
            answer = 2;
            nodding_coordinates.clear();
            shaking_coordinates.clear();
          }

          // classify as shake only when having significant bidirectional horizontal movement and direction change and no nodding movements
          else if (shakeScan > 0 && nodScan == 0 && abs(max_shake - min_shake) <= HORIZONTAL_ADJUSTMENT * distance_adjustment)
          {
            std::cout << "NO\n";
            answer = 1;
            nodding_coordinates.clear();
            shaking_coordinates.clear();
          }
          else if (nodScan || shakeScan ||
                   abs(max_shake - min_shake) > HORIZONTAL_ADJUSTMENT * distance_adjustment ||
                   abs(max_nod - min_nod) > VERTICAL_ADJUSTMENT * distance_adjustment)
          {
            std::cout << "INVALID\n";
            answer = 3;
            nodding_coordinates.clear();
            shaking_coordinates.clear();
          }
        }
      }
    }

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;

    if (!poller.Next(&packet))
      break;

    auto &output_frame = packet.Get<mediapipe::ImageFrame>();

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

    // anotate question on top and display window
    cv::putText(output_frame_mat, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0, 255, 0), 2);
    cv::putText(output_frame_mat, text2, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0, 255, 0), 2);
    cv::Mat dst;
    cv::resize(output_frame_mat, dst, cv::Size(960, 720), cv::INTER_LINEAR);
    cv::imshow(kWindowName, dst);
    cv::resizeWindow(kWindowName, 960, 720);

    // Press any key to exit.
    const int pressed_key = cv::waitKey(5);
    if (pressed_key >= 0 && pressed_key != 255)
      grab_frames = false;

    // assess frame processing time and readjust in realtime analysis frame based on system speed
    if (time_index == 10)
    {
      auto time_delta = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - prev_time);
      prev_time = std::chrono::high_resolution_clock::now();
      time_index = 0;
      framesToAnalyze = int(500000 * 8 / time_delta.count());
      // std::cout << framesToAnalyze << std::endl;
    }

    // question control logic and logging
    if (answer == 3 && ready_next)
    {
      wait_interval = std::chrono::high_resolution_clock::now();
      ready_next = false;
      text = "Invalid gesture, you can try again though!";
      text2 = "Please nod down for yes or shake for no.";
    }

    else if (answer > 0 && ready_next)
    {
      wait_interval = std::chrono::high_resolution_clock::now();
      ready_next = false;
      if (answer == 1)
        text = "NO";
      else
        text = "YES";

      // logging to file
      auto now = std::chrono::system_clock::now();
      auto in_time_t = std::chrono::system_clock::to_time_t(now);
      std::tm bt = *std::localtime(&in_time_t);
      std::ostringstream oss;
      oss << std::put_time(&bt, "%Y-%m-%d_%H-%M-%S");
      std::string str_time = oss.str();
      std::string line = questions[q_i] + ',' + text + ',' + str_time;
      std::ofstream file(filename, std::ios::app);
      if (!file.is_open())
      {
        std::cout << "Could not open the file for appending." << std::endl;
      }
      file << line << std::endl;
      std::cout << filename << std::endl;
      std::cout << line << std::endl;
      file.close();

      if (q_i + 1 < q_max)
        q_i += 1;
      else
        q_i = 0;
    }

    // wait 1.5s before showing next question
    auto timePassed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - wait_interval);
    if (timePassed.count() > 1500)
    {
      ready_next = true;
      text = questions[q_i];
      text2 = "";
    }
  }

  ABSL_LOG(INFO) << "Shutting down.";

  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok())
  {
    ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  }
  else
  {
    ABSL_LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
