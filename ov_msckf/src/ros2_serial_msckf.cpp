/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <memory>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "ros/ROS2Visualizer.h"
#include "utils/dataset_reader.h"

#include <chrono>

using namespace ov_msckf;

/**
 * Decompress a CompressedImage message to an Image message
 * Supports JPEG, PNG, and other OpenCV-compatible formats
 */
std::shared_ptr<sensor_msgs::msg::Image> decompress_image(
    const sensor_msgs::msg::CompressedImage& compressed_msg) {
  
  try {
    // Decode the compressed image data
    cv::Mat decompressed = cv::imdecode(
        cv::Mat(1, compressed_msg.data.size(), CV_8UC1, 
                const_cast<uint8_t*>(compressed_msg.data.data())),
        cv::IMREAD_COLOR);
    
    if (decompressed.empty()) {
      PRINT_WARNING("[DECOMPRESS]: Failed to decompress image\n");
      auto empty_img = std::make_shared<sensor_msgs::msg::Image>();
      empty_img->header = compressed_msg.header;
      return empty_img;
    }
    
    // Convert OpenCV mat to ROS2 Image message
    auto img_bridge = cv_bridge::CvImage(
        compressed_msg.header, "bgr8", decompressed);
    
    return std::make_shared<sensor_msgs::msg::Image>(
        *img_bridge.toImageMsg());
    
  } catch (const std::exception& e) {
    PRINT_ERROR("[DECOMPRESS]: Exception during decompression: %s\n", e.what());
    auto empty_img = std::make_shared<sensor_msgs::msg::Image>();
    empty_img->header = compressed_msg.header;
    return empty_img;
  }
}

// Main function
int main(int argc, char **argv) {
  auto total_start = std::chrono::high_resolution_clock::now();

  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  // Initialize ROS2
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("ros2_serial_msckf");

  // Get parameters
  node->declare_parameter("config_path", config_path);
  config_path = node->get_parameter("config_path").as_string();

  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  // Note: ROS2 doesn't use set_node_handler - parameters are handled directly
  parser->set_node(node);

  // Verbosity
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Create our VIO system
  VioManagerOptions params;
  params.print_and_load(parser);
  // params.num_opencv_threads = 0; // uncomment if you want repeatability
  // params.use_multi_threading_pubs = 0; // uncomment if you want repeatability
  params.use_multi_threading_subs = false;

  std::shared_ptr<VioManager> sys = std::make_shared<VioManager>(params);
  std::shared_ptr<ROS2Visualizer> viz = std::make_shared<ROS2Visualizer>(node, sys);

  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "[SERIAL]: unable to parse all parameters, please fix\n" RESET);
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Our imu topic
  std::string topic_imu;
  node->declare_parameter("topic_imu", "/imu0");
  topic_imu = node->get_parameter("topic_imu").as_string();
  parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
  PRINT_INFO("[SERIAL]: imu: %s\n", topic_imu.c_str());

  // Our camera topics
  std::vector<std::string> topic_cameras;
  for (int i = 0; i < params.state_options.num_cameras; i++) {
    std::string cam_topic_param = "topic_camera" + std::to_string(i);
    std::string cam_topic_default = "/cam" + std::to_string(i) + "/image_raw";
    node->declare_parameter(cam_topic_param, cam_topic_default);
    std::string cam_topic = node->get_parameter(cam_topic_param).as_string();
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);
    cam_topic = "/uav1/bluefox/image_raw";
    topic_cameras.emplace_back(cam_topic);
    PRINT_INFO("[SERIAL]: cam: %s\n", cam_topic.c_str());
  }

  // Location of the ROS bag we want to read in
  std::string path_to_bag;
  node->declare_parameter("path_bag", "/home/patrick/datasets/eth/V1_01_easy");
  path_to_bag = node->get_parameter("path_bag").as_string();
  PRINT_INFO("[SERIAL]: ros bag path is: %s\n", path_to_bag.c_str());

  // Load groundtruth if we have it
  // NOTE: needs to be a csv ASL format file
  std::map<double, Eigen::Matrix<double, 17, 1>> gt_states;
  if (node->has_parameter("path_gt")) {
    node->declare_parameter("path_gt", "");
    std::string path_to_gt = node->get_parameter("path_gt").as_string();
    if (!path_to_gt.empty()) {
      ov_core::DatasetReader::load_gt_file(path_to_gt, gt_states);
      PRINT_INFO("[SERIAL]: gt file path is: %s\n", path_to_gt.c_str());
    }
  }

  // Get our start location and how much of the bag we want to play
  // Make the bag duration < 0 to just process to the end of the bag
  double bag_start, bag_durr;
  node->declare_parameter("bag_start", 0.0);
  node->declare_parameter("bag_durr", -1.0);
  bag_start = node->get_parameter("bag_start").as_double();
  bag_durr = node->get_parameter("bag_durr").as_double();
  PRINT_INFO("[SERIAL]: bag start: %.1f\n", bag_start);
  PRINT_INFO("[SERIAL]: bag duration: %.1f\n", bag_durr);

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Load rosbag2 here, and find messages we can play
  rosbag2_storage::StorageOptions storage_options{};
  storage_options.uri = path_to_bag;
  // Use sqlite3 as default (most common rosbag2 format)
  // If you get warnings about mcap, this is correct
  storage_options.storage_id = "mcap";

  rosbag2_cpp::ConverterOptions converter_options{};
  converter_options.input_serialization_format = "cdr";

  rosbag2_cpp::Reader reader;
  try {
    reader.open(storage_options, converter_options);
  } catch (const std::exception& e) {
    PRINT_ERROR(RED "[SERIAL]: Failed to open rosbag at %s\n" RESET, path_to_bag.c_str());
    PRINT_ERROR(RED "[SERIAL]: Error: %s\n" RESET, e.what());
    PRINT_ERROR(RED "[SERIAL]: Make sure the path is correct and the bag is not corrupted\n" RESET);
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }

  // Get bag metadata to determine time range
  auto metadata = reader.get_metadata();
  
  // Validate metadata timestamp is reasonable (should be between 2000 and 2100)
  auto start_time_ns = metadata.starting_time.time_since_epoch().count();
  if (start_time_ns > 4e18 || start_time_ns < 1e18) {
    PRINT_WARNING("[SERIAL]: Metadata timestamp may be invalid (%.0f ns), using first message\n", 
               (double)start_time_ns);
    start_time_ns = 0;
  }
  
  rclcpp::Time time_init = rclcpp::Time(start_time_ns);
  time_init += rclcpp::Duration(std::chrono::duration<double>(bag_start));

  rclcpp::Time time_finish;
  if (bag_durr < 0) {
    time_finish = rclcpp::Time(start_time_ns + 
                               std::chrono::nanoseconds(metadata.duration).count());
  } else {
    time_finish = time_init + rclcpp::Duration(std::chrono::duration<double>(bag_durr));
  }

  PRINT_INFO("[SERIAL]: Bag duration: %.1f seconds\n", metadata.duration / 1e9);
  PRINT_INFO("[SERIAL]: time start = %.6f\n", time_init.seconds());
  PRINT_INFO("[SERIAL]: time end   = %.6f\n", time_finish.seconds());
  PRINT_INFO("[SERIAL]: Looking for IMU topic: %s\n", topic_imu.c_str());
  for (int i = 0; i < params.state_options.num_cameras; i++) {
    PRINT_INFO("[SERIAL]: Looking for camera %d topic: %s\n", i, topic_cameras.at(i).c_str());
  }

  // We going to loop through and collect a list of all messages
  // This is done so we can access arbitrary points in the bag
  // NOTE: rosbag2 doesn't support seeking like rosbag1, so we read sequentially
  // but we still store messages to allow access to arbitrary timesteps
  double max_camera_time = -1;
  std::vector<rosbag2_storage::SerializedBagMessage> msgs;
  std::vector<std::string> msgs_topics;

  while (reader.has_next()) {
    auto bag_message = reader.read_next();
    
    // Check if message is within time range
    rclcpp::Time msg_time = rclcpp::Time(bag_message->recv_timestamp);
    if (msg_time < time_init)
      continue;
    if (msg_time > time_finish)
      break;

    // Check if this is an IMU topic
    if (bag_message->topic_name == topic_imu) {
      msgs.push_back(*bag_message);
      msgs_topics.push_back(bag_message->topic_name);
      continue;
    }

    // Check if this is a camera topic
    for (int i = 0; i < params.state_options.num_cameras; i++) {
      if (bag_message->topic_name == topic_cameras.at(i)) {
        msgs.push_back(*bag_message);
        msgs_topics.push_back(bag_message->topic_name);
        max_camera_time = std::max(max_camera_time, msg_time.seconds());
        break;
      }
    }
  }

  // Check to make sure we have data to play
  if (msgs.empty()) {
    PRINT_ERROR(RED "[SERIAL]: No messages to play on specified topics.  Exiting.\n" RESET);
    PRINT_ERROR(RED "[SERIAL]: This could mean:\n" RESET);
    PRINT_ERROR(RED "[SERIAL]:   1. Topic names don't match bag contents\n" RESET);
    PRINT_ERROR(RED "[SERIAL]:   2. Time range is incorrect (start=%.1f, duration=%.1f)\n" RESET, bag_start, bag_durr);
    PRINT_ERROR(RED "[SERIAL]:   3. Rosbag file is corrupted or empty\n" RESET);
    PRINT_ERROR(RED "[SERIAL]: Expected topics:\n" RESET);
    PRINT_ERROR(RED "[SERIAL]:   IMU: %s\n" RESET, topic_imu.c_str());
    for (int i = 0; i < params.state_options.num_cameras; i++) {
      PRINT_ERROR(RED "[SERIAL]:   Camera %d: %s\n" RESET, i, topic_cameras.at(i).c_str());
    }
    PRINT_ERROR(RED "[SERIAL]: Use 'ros2 bag info <bag_path>' to check bag contents\n" RESET);
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
     
  PRINT_INFO("[SERIAL]: total of %zu messages!\n", msgs.size()); 

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Need to create a serialization object for deserialization
  // Note: rosbag2 uses rcutils_uint8_array which needs to be wrapped in SerializedMessage
  
  // Reopen reader to iterate through messages again for processing
  rosbag2_cpp::Reader reader2;
  reader2.open(storage_options, converter_options);

  // Loop through our message array, and lets process them
  std::set<int> used_index;
  int msg_idx = 0;

  while (reader2.has_next() && msg_idx < (int)msgs.size()) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<clock::time_point> tp;
    std::vector<std::string> tp_name;

    tp.push_back(clock::now());
    tp_name.push_back("1");

    auto bag_message = reader2.read_next();
    rclcpp::Time msg_time = rclcpp::Time(bag_message->recv_timestamp);

    tp.push_back(clock::now());
    tp_name.push_back("2");

    // Skip messages outside our time range
    if (msg_time < time_init) {
      continue;
    }
    if (msg_time > time_finish || msg_time.seconds() > max_camera_time) {
      break;
    }

    // Skip messages that we have already used
    if (used_index.find(msg_idx) != used_index.end()) {
      used_index.erase(msg_idx);
      msg_idx++;
      continue;
    }

    tp.push_back(clock::now());
    tp_name.push_back("3");

    // IMU processing
    if (bag_message->topic_name == topic_imu) {
      // Deserialize the message using ROS2 serialization
      sensor_msgs::msg::Imu imu_msg;
      rclcpp::Serialization<sensor_msgs::msg::Imu> serialization;
      
      // Create a SerializedMessage from the raw data
      rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
      serialization.deserialize_message(&serialized_msg, &imu_msg);
      
      PRINT_INFO("processing imu = %.3f sec\n", msg_time.seconds() - time_init.seconds());
      auto start = std::chrono::high_resolution_clock::now();
      viz->callback_inertial(std::make_shared<sensor_msgs::msg::Imu>(imu_msg));
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      PRINT_INFO("imu proc time: %.3f sec\n", duration.count());
      msg_idx++;
      continue;
    }

    tp.push_back(clock::now());
    tp_name.push_back("4");

    // Camera processing
    for (int cam_id = 0; cam_id < params.state_options.num_cameras; cam_id++) {

      tp.push_back(clock::now());
      tp_name.push_back("5");

      // Skip if this message is not a camera topic
      if (bag_message->topic_name != topic_cameras.at(cam_id))
        continue;

      // We have a matching camera topic here, now find the other cameras for this time
      // For each camera, we will find the nearest timestamp (within 0.02sec) that is greater than the current
      // If we are unable, then this message should just be skipped since it isn't a sync'ed pair!
      std::map<int, int> camid_to_msg_index;
      double meas_time = msg_time.seconds();
      
      for (int cam_idt = 0; cam_idt < params.state_options.num_cameras; cam_idt++) {
        if (cam_idt == cam_id) {
          camid_to_msg_index.insert({cam_id, msg_idx});
          continue;
        }
        
        int cam_idt_idx = -1;
        for (int mt = msg_idx; mt < (int)msgs.size(); mt++) {
          if (msgs_topics.at(mt) != topic_cameras.at(cam_idt))
            continue;
          rclcpp::Time check_time = rclcpp::Time(msgs.at(mt).recv_timestamp);
          if (std::abs(check_time.seconds() - meas_time) < 0.02)
            cam_idt_idx = mt;
          break;
        }
        
        if (cam_idt_idx != -1) {
          camid_to_msg_index.insert({cam_idt, cam_idt_idx});
        }
      }

      tp.push_back(clock::now());
      tp_name.push_back("6");

      // Skip processing if we were unable to find any messages
      if ((int)camid_to_msg_index.size() != params.state_options.num_cameras) {
        PRINT_INFO(YELLOW "[SERIAL]: Unable to find stereo pair for message %d at %.2f into bag (will skip!)\n" RESET,
                    msg_idx, meas_time - time_init.seconds());
        msg_idx++;
        break;
      }

      // Check if we should initialize using the groundtruth
      Eigen::Matrix<double, 17, 1> imustate;
      if (!gt_states.empty() && !sys->initialized() &&
          ov_core::DatasetReader::get_gt_state(meas_time, imustate, gt_states)) {
        // biases are pretty bad normally, so zero them
        // imustate.block(11,0,6,1).setZero();
        sys->initialize_with_gt(imustate);
      }

      // Deserialize camera messages (supports both compressed and uncompressed)
      rclcpp::Serialization<sensor_msgs::msg::Image> image_serialization;
      rclcpp::Serialization<sensor_msgs::msg::CompressedImage> compressed_serialization;
      std::vector<std::shared_ptr<sensor_msgs::msg::Image>> images;

      tp.push_back(clock::now());
      tp_name.push_back("7");

      for (int i = 0; i < params.state_options.num_cameras; i++) {
        rclcpp::SerializedMessage serialized_msg(*msgs.at(camid_to_msg_index.at(i)).serialized_data);
        const std::string& topic_name = msgs_topics.at(camid_to_msg_index.at(i));
        std::shared_ptr<sensor_msgs::msg::Image> img_ptr;

        tp.push_back(clock::now());
        tp_name.push_back("8");
        
        // Check if this is a compressed image topic
        bool is_compressed = (topic_name.find("/compressed") != std::string::npos);
        
        if (is_compressed) {
          tp.push_back(clock::now());
          tp_name.push_back("7");
          // Deserialize and decompress CompressedImage
          sensor_msgs::msg::CompressedImage compressed_msg;
          compressed_serialization.deserialize_message(&serialized_msg, &compressed_msg);
          img_ptr = decompress_image(compressed_msg);
          PRINT_INFO("[SERIAL]: Decompressed image from %s\n", topic_name.c_str());
          tp.push_back(clock::now());
          tp_name.push_back("8");
        } else {
          tp.push_back(clock::now());
          tp_name.push_back("9");
          // Deserialize as regular Image
          sensor_msgs::msg::Image img_msg;
          image_serialization.deserialize_message(&serialized_msg, &img_msg);
          img_ptr = std::make_shared<sensor_msgs::msg::Image>(img_msg);
          tp.push_back(clock::now());
          tp_name.push_back("10");
        }
        
        images.push_back(img_ptr);
      }

      tp.push_back(clock::now());
      tp_name.push_back("11");

      // Pass our data into our visualizer callbacks!
      PRINT_INFO("processing cam = %.3f sec\n", msg_time.seconds() - time_init.seconds());
      if (params.state_options.num_cameras == 1) {
        auto start = std::chrono::high_resolution_clock::now();
        viz->callback_monocular(images.at(0), 0);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        PRINT_INFO("img proc time: %.3f sec\n", duration.count());
      } else if (params.state_options.num_cameras == 2) {
        used_index.insert(camid_to_msg_index.at(0)); // skip this message
        used_index.insert(camid_to_msg_index.at(1)); // skip this message
        viz->callback_stereo(images.at(0), images.at(1), 0, 1);
      } else {
        PRINT_ERROR(RED "[SERIAL]: We currently only support 1 or 2 camera serial input....\n" RESET);
        rclcpp::shutdown();
        return EXIT_FAILURE;
      }

      tp.push_back(clock::now());
      tp_name.push_back("12");

      double total_time = 0.0;
      for(unsigned int i = 1; i < tp.size(); i++){
        double duration = ((double)std::chrono::duration_cast<std::chrono::nanoseconds>(tp.at(i) - tp.at(i-1)).count())/1000000000.0;
        total_time += duration;
        PRINT_INFO("%s: %f msec\n", tp_name.at(i-1).c_str(), duration);
      }
      PRINT_INFO("total: %f msec\n", total_time);

      msg_idx++;
      break;
    }
  }

  // Final visualization
  viz->visualize_final();

  // Shutdown ROS2
  rclcpp::shutdown();

  auto total_end = std::chrono::high_resolution_clock::now();
  double total_duration = ((double)std::chrono::duration_cast<std::chrono::nanoseconds>(total_end - total_start).count())/1000000000.0;
  PRINT_INFO("TOTAL TIME: %.3f sec\n", total_duration);

  // Done!
  return EXIT_SUCCESS;
}