#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "tracking_msgs/msg/detections3_d.hpp"
#include "tracking_msgs/msg/detection3_d.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

#include "ar_track_alvar_msgs/msg/alvar_markers.hpp"

using std::placeholders::_1;

class ARPreProc : public rclcpp::Node
{
  public:
    ARPreProc()
    : Node("ar_preproc_node")
    {
      this->subscription_ = this->create_subscription<ar_track_alvar_msgs::msg::AlvarMarkers>(
      "ar_pose_detections", 10, std::bind(&ARPreProc::topic_callback, this, _1));
      this->publisher_ = this->create_publisher<tracking_msgs::msg::Detections3D>("converted_detections", 10);

      this->declare_parameter("tracker_frame",rclcpp::ParameterType::PARAMETER_STRING);
      this->declare_parameter("ar_tag_ids",rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY);
      this->declare_parameter("labels",rclcpp::ParameterType::PARAMETER_STRING_ARRAY);
      
      this->tracker_frame_ = this->get_parameter("tracker_frame").as_string();
      this->ar_tag_ids_ = this->get_parameter("ar_tag_ids").as_integer_array();
      this->labels_ = this->get_parameter("labels").as_string_array();

      this->tf_buffer_ =std::make_unique<tf2_ros::Buffer>(this->get_clock());
      this->tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

      this->det_msg_ = tracking_msgs::msg::Detection3D();
      this->dets_msg_ = tracking_msgs::msg::Detections3D();
      this->dets_msg_.detections.reserve(this->max_dets_);
    }

  private:
    void topic_callback(const ar_track_alvar_msgs::msg::AlvarMarkers::SharedPtr msg)
    {
        rclcpp::Time time_det_rcvd = this->get_clock()->now();
        diagnostic_msgs::msg::KeyValue kv;

        this->dets_msg_ = tracking_msgs::msg::Detections3D();
        this->dets_msg_.header.stamp = msg->header.stamp;     
        this->dets_msg_.header.frame_id = this->tracker_frame_;

        this->det_msg_ = tracking_msgs::msg::Detection3D();

        // Add metadata for later analysis
        kv.key = "time_det_rcvd";
        kv.value = std::to_string(time_det_rcvd.nanoseconds());
        this->dets_msg_.metadata.emplace_back(kv);
        kv.key = "num_dets_rcvd";
        kv.value = std::to_string(1);
        this->dets_msg_.metadata.emplace_back(kv);

        for (auto it = msg->markers.begin(); it != msg->markers.end(); it++)
        {
            auto found = std::find(this->ar_tag_ids_.begin(),this->ar_tag_ids_.end(),it->id);

            if (found==this->ar_tag_ids_.end()) continue;

            uint8_t idx = found - this->ar_tag_ids_.begin();

            // Convert spatial information
            this->obj_pose_det_frame_ = geometry_msgs::msg::PoseStamped();
            this->obj_pose_det_frame_.header = msg->header;
            this->obj_pose_det_frame_.pose = it->pose.pose;
            this->obj_pose_trk_frame_ = this->tf_buffer_->transform(this->obj_pose_det_frame_,this->tracker_frame_);
            this->det_msg_.pose = obj_pose_trk_frame_.pose;
            this->det_msg_.bbox.center = obj_pose_trk_frame_.pose;
            this->det_msg_.bbox.size.x = 0;
            this->det_msg_.bbox.size.y = 0;
            this->det_msg_.bbox.size.z = 0;

            // Add semantic information
            this->det_msg_.class_string = this->labels_[idx];
            this->det_msg_.class_confidence = .99;
            this->dets_msg_.detections.emplace_back(this->det_msg_);
        }
        this->publisher_->publish(this->dets_msg_);
           
    }

    std::string tracker_frame_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    geometry_msgs::msg::PoseStamped obj_pose_det_frame_;
    geometry_msgs::msg::PoseStamped obj_pose_trk_frame_;

    rclcpp::Subscription<ar_track_alvar_msgs::msg::AlvarMarkers>::SharedPtr subscription_;
    rclcpp::Publisher<tracking_msgs::msg::Detections3D>::SharedPtr publisher_;
    tracking_msgs::msg::Detections3D dets_msg_;
    tracking_msgs::msg::Detection3D det_msg_;
    int max_dets_{250}; 

    std::vector<long int> ar_tag_ids_;
    std::vector<std::string> labels_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ARPreProc>());
  rclcpp::shutdown();
  return 0;
}