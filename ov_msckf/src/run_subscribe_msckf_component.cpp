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

#include <memory>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/dataset_reader.h"

#include "ros/ROS2Visualizer.h"
#include <rclcpp/rclcpp.hpp>

using namespace ov_msckf;

namespace msckf_component {

class SubscribeMSCKF : public rclcpp::Node {
    public:
        SubscribeMSCKF(const rclcpp::NodeOptions & options) : rclcpp::Node("run_subscribe_msckf", options) {
            //options.allow_undeclared_parameters(true);
            //options.automatically_declare_parameters_from_overrides(true);
            timer_initialization_ = create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&SubscribeMSCKF::timerInitialization, this));
            //init();
        }

        ~SubscribeMSCKF(){
            viz->visualize_final();
            rclcpp::shutdown();
        }

    private:
        rclcpp::TimerBase::SharedPtr timer_initialization_;
        bool is_initialized_;

        std::shared_ptr<VioManager> sys;
        std::shared_ptr<ROS2Visualizer> viz;

        void timerInitialization() {
            declare_parameter<std::string>("config_path");
            std::string config_path;
            get_parameter<std::string>("config_path", config_path);
            RCLCPP_INFO(get_logger(), "CONFIG PATH: %s", config_path.c_str());

            declare_parameter<std::string>("frames_prefix", "");
            std::string frames_prefix;
            get_parameter<std::string>("frames_prefix", frames_prefix);
            RCLCPP_INFO(get_logger(), "FRAMES PREFIX: %s", frames_prefix.c_str());

            declare_parameter<std::string>("global_frame_name", "global");
            std::string global_frame_name;
            get_parameter<std::string>("global_frame_name", global_frame_name);
            RCLCPP_INFO(get_logger(), "GLOBAL FRAME NAME: %s", global_frame_name.c_str());

            declare_parameter<std::string>("imu_frame_name", "imu");
            std::string imu_frame_name;
            get_parameter<std::string>("imu_frame_name", imu_frame_name);
            RCLCPP_INFO(get_logger(), "IMU FRAME NAME: %s", imu_frame_name.c_str());

            declare_parameter<std::string>("cam_frame_name", "cam0");
            std::string cam_frame_name;
            get_parameter<std::string>("cam_frame_name", cam_frame_name);
            RCLCPP_INFO(get_logger(), "CAM FRAME NAME: %s", cam_frame_name.c_str());

            // Load the config
            auto parser = std::make_shared<ov_core::YamlParser>(config_path);
            std::shared_ptr<rclcpp::Node> node = shared_from_this();
            parser->set_node(node);

            // Verbosity
            std::string verbosity = "DEBUG";
            parser->parse_config("verbosity", verbosity);
            ov_core::Printer::setPrintLevel(verbosity);

            // Create our VIO system
            VioManagerOptions params;
            params.print_and_load(parser);
            params.use_multi_threading_subs = true;
            sys = std::make_shared<VioManager>(params);
            viz = std::make_shared<ROS2Visualizer>(node, sys, nullptr, frames_prefix, global_frame_name, imu_frame_name, cam_frame_name);
            viz->setup_subscribers(parser);

            // Ensure we read in all parameters required
            if (!parser->successful()) {
                PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
                RCLCPP_ERROR(get_logger(), "unable to parse all parameters, please fix");
                std::exit(EXIT_FAILURE);
            }

            is_initialized_ = true;
            timer_initialization_->cancel();
        }
};

}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(msckf_component::SubscribeMSCKF)