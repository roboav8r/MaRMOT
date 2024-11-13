#include <chrono>
#include <memory>
#include <cv_bridge/cv_bridge.h> // cv_bridge converts between ROS 2 image messages and OpenCV image representations.
#include <image_transport/image_transport.hpp> // Using image_transport allows us to publish and subscribe to compressed image streams in ROS2
#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "tracking_msgs/msg/detection3_d.hpp"
#include "tracking_msgs/msg/detections3_d.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"


using std::placeholders::_1;
using namespace std::chrono_literals;

class DepthAIPreProc : public rclcpp::Node
{
  public:
    DepthAIPreProc()
    : Node("depthai_preproc_node"), 
    nh_(std::shared_ptr<DepthAIPreProc>(this, [](auto *) {})),
    image_transport_(nh_)
    {
      subscription_ = this->create_subscription<vision_msgs::msg::Detection3DArray>("depthai_detections", 10, std::bind(&DepthAIPreProc::detection_callback, this, _1));
      image_sub_ = image_transport_.subscribe("depthai_img", 10, std::bind(&DepthAIPreProc::image_callback, this, std::placeholders::_1));
      publisher_ = this->create_publisher<tracking_msgs::msg::Detections3D>("converted_detections", 10);

      declare_parameter("detector_frame",rclcpp::ParameterType::PARAMETER_STRING);
      declare_parameter("tracker_frame",rclcpp::ParameterType::PARAMETER_STRING);
      declare_parameter("labels",rclcpp::ParameterType::PARAMETER_STRING_ARRAY);
      declare_parameter("nn_img_size", rclcpp::ParameterType::PARAMETER_INTEGER);
      declare_parameter("x_correction_factor", rclcpp::ParameterType::PARAMETER_DOUBLE);
      declare_parameter("y_correction_factor", rclcpp::ParameterType::PARAMETER_DOUBLE);

      this->detector_frame_ = get_parameter("detector_frame").as_string();
      this->tracker_frame_ = get_parameter("tracker_frame").as_string();
      this->labels_ = get_parameter("labels").as_string_array();
      this->nn_img_size = get_parameter("nn_img_size").as_int();
      this->x_corr_factor_ = get_parameter("x_correction_factor").as_double();
      this->y_corr_factor_ = get_parameter("y_correction_factor").as_double();

      this->tf_buffer_ =std::make_unique<tf2_ros::Buffer>(this->get_clock(),500ms);
      this->tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

      // Block until transform becomes available
      this->tf_buffer_->canTransform(this->detector_frame_,this->tracker_frame_,tf2::TimePointZero, 30s);
      this->det_msg_ = tracking_msgs::msg::Detection3D();
      this->dets_msg_ = tracking_msgs::msg::Detections3D();
      this->dets_msg_.detections.reserve(this->max_dets_);
    }

  private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        this->img_rcvd_ = true;

        cv_in_ = cv_bridge::toCvShare(msg, "bgr8");
    }
    
    void detection_callback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
      rclcpp::Time time_det_rcvd = this->get_clock()->now();
      diagnostic_msgs::msg::KeyValue kv;     

      this->dets_msg_ = tracking_msgs::msg::Detections3D();
      this->dets_msg_.header.stamp = msg->header.stamp;     
      this->dets_msg_.header.frame_id = this->tracker_frame_;

      // Add metadata for later analysis
      kv.key = "time_det_rcvd";
      kv.value = std::to_string(time_det_rcvd.nanoseconds());
      this->dets_msg_.metadata.emplace_back(kv);
      kv.key = "num_dets_rcvd";
      kv.value = std::to_string(msg->detections.size());
      this->dets_msg_.metadata.emplace_back(kv);

      for (auto it = msg->detections.begin(); it != msg->detections.end(); it++)
      {
            this->det_msg_ = tracking_msgs::msg::Detection3D();

            // Convert spatial information
            this->pose_det_ = geometry_msgs::msg::PoseStamped();
            this->pose_det_.header = msg->header;
            this->pose_det_.pose = it->results[0].pose.pose;
            this->pose_det_.pose.position.y *= -1; // Fix OAK-D y-axis inversion / lefthand coord frame
            this->tf_buffer_->transform(this->pose_det_,this->pose_tracker_,this->tracker_frame_,tf2::TimePointZero,this->detector_frame_,200ms);
            this->pose_tracker_.pose.position.x *= this->x_corr_factor_;
            this->pose_tracker_.pose.position.y *= this->y_corr_factor_;
            this->det_msg_.pose = this->pose_tracker_.pose;
            this->det_msg_.bbox.center = this->pose_tracker_.pose;
            this->det_msg_.bbox.size.x = 0;
            this->det_msg_.bbox.size.y = 0;
            this->det_msg_.bbox.size.z = 0;

            // Add semantic information
            this->det_msg_.class_string = this->labels_[std::stoi(it->results[0].hypothesis.class_id)];
            this->det_msg_.class_confidence = it->results[0].hypothesis.score;

            // Add image data
            if (img_rcvd_) {

                // Generate mask, mask cv_in and publish
                // Compute mask pixel locations
                float scale = (float)std::min(cv_in_->image.rows, cv_in_->image.cols)/(float)nn_img_size;
                int x_off = (cv_in_->image.cols - std::min(cv_in_->image.rows, cv_in_->image.cols))/2;
                int y_off = (cv_in_->image.rows - std::min(cv_in_->image.rows, cv_in_->image.cols))/2;

                int xmin = x_off + std::max(0, (int)(scale*(it->bbox.center.position.x - (it->bbox.size.x)/2)));
                int ymin = y_off + std::max(0, (int)(scale*(it->bbox.center.position.y - (it->bbox.size.y)/2)));
                int xmax = x_off + std::min(std::min(cv_in_->image.rows, cv_in_->image.cols), (int)(scale*(it->bbox.center.position.x + (it->bbox.size.x)/2)));
                int ymax = y_off + std::min(std::min(cv_in_->image.rows, cv_in_->image.cols), (int)(scale*(it->bbox.center.position.y + (it->bbox.size.y)/2)));

                cv::Mat cv_out_mat = cv_in_->image(cv::Range(ymin, ymax),cv::Range(xmin, xmax));

                cv_out_.header = cv_in_->header;
                cv_out_.encoding = cv_in_->encoding;
                cv_out_.image = cv_out_mat;

                img_out_ = cv_out_.toImageMsg();
                det_msg_.image = *img_out_;
                det_msg_.image_available = true;
            }

            // Add detection message to detections message
            this->dets_msg_.detections.emplace_back(this->det_msg_);
      }

      this->publisher_->publish(this->dets_msg_);
           
    }

    std::string detector_frame_;
    std::string tracker_frame_;
    geometry_msgs::msg::PoseStamped pose_det_;
    geometry_msgs::msg::PoseStamped pose_tracker_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr subscription_;
    rclcpp::Publisher<tracking_msgs::msg::Detections3D>::SharedPtr publisher_;
    tracking_msgs::msg::Detections3D dets_msg_;
    tracking_msgs::msg::Detection3D det_msg_;
    int max_dets_{250}; 

    std::vector<std::string> labels_;
    double x_corr_factor_;
    double y_corr_factor_;

    // OpenCV / vision processing
    int nn_img_size;
    rclcpp::Node::SharedPtr nh_;
    bool img_rcvd_{false};
    image_transport::ImageTransport image_transport_;
    image_transport::Subscriber image_sub_;
    cv_bridge::CvImageConstPtr cv_in_; // Received from camera ROS msg, converted to CV pointer
    cv_bridge::CvImage cv_out_; // CV pointer, to be output as a ROS image
    sensor_msgs::msg::Image::SharedPtr img_out_; // ROS image output pointer
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DepthAIPreProc>());
  rclcpp::shutdown();
  return 0;
}