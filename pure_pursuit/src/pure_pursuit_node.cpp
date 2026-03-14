#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

using namespace std;

struct Waypoint {
    double x, y;
};

class PurePursuit : public rclcpp::Node
{
private:
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

    double lookahead_dist_;
    double wheelbase_;
    double speed_;
    double max_steer_;
    string waypoint_file_;

    vector<Waypoint> waypoints_;
    int last_closest_idx_{0};

public:
    PurePursuit() : Node("pure_pursuit_node")
    {
        this->declare_parameter("lookahead_dist",  1.5);
        this->declare_parameter("wheelbase",       0.33);
        this->declare_parameter("speed",           1.5);
        this->declare_parameter("max_steer_angle", 0.4189);
        this->declare_parameter("waypoint_file",   "waypoints.csv");
        this->declare_parameter("pose_topic",      "/pf/viz/inferred_pose");
        this->declare_parameter("drive_topic",     "/drive");

        lookahead_dist_ = this->get_parameter("lookahead_dist").as_double();
        wheelbase_      = this->get_parameter("wheelbase").as_double();
        speed_          = this->get_parameter("speed").as_double();
        max_steer_      = this->get_parameter("max_steer_angle").as_double();
        waypoint_file_  = this->get_parameter("waypoint_file").as_string();

        load_waypoints(waypoint_file_);

        drive_pub_  = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
                          this->get_parameter("drive_topic").as_string(), 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
                          "/pure_pursuit/goal_marker", 10);

        pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            this->get_parameter("pose_topic").as_string(), 10,
            std::bind(&PurePursuit::pose_callback, this, std::placeholders::_1));
    }

    void load_waypoints(const string &filepath)
    {
        ifstream file(filepath);
        if (!file.is_open()) return;
        string line;
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            istringstream ss(line);
            string tok;
            Waypoint wp;
            getline(ss, tok, ','); wp.x = stod(tok);
            getline(ss, tok, ','); wp.y = stod(tok);
            waypoints_.push_back(wp);
        }
    }

    void pose_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
    {
        if (waypoints_.empty()) return;

        // 1. Current State
        double cx = pose_msg->pose.pose.position.x;
        double cy = pose_msg->pose.pose.position.y;
        tf2::Quaternion q(
            pose_msg->pose.pose.orientation.x,
            pose_msg->pose.pose.orientation.y,
            pose_msg->pose.pose.orientation.z,
            pose_msg->pose.pose.orientation.w);
        tf2::Matrix3x3 mat(q);
        double r, p, yaw;
        mat.getRPY(r, p, yaw);

        // 2. Find Closest & Goal Points
        int closest_idx = find_closest_index(cx, cy);
        int goal_idx_primary;
        Waypoint goal = find_goal_point(cx, cy, closest_idx, goal_idx_primary);

        // 3. Spline Generation Logic
        int next_idx = (goal_idx_primary + 1) % waypoints_.size();
        double goal_yaw = atan2(waypoints_[next_idx].y - goal.y, waypoints_[next_idx].x - goal.x);

        double dist = distance(cx, cy, goal.x, goal.y);
        double handle_len = dist * 0.5; 

        double p1x = cx + cos(yaw) * handle_len;
        double p1y = cy + sin(yaw) * handle_len;
        double p2x = goal.x - cos(goal_yaw) * handle_len;
        double p2y = goal.y - sin(goal_yaw) * handle_len;

        // t=0.15 makes the car more reactive/tighter on turn-in
        double t = 0.15; 
        double invT = 1.0 - t;
        double tx = pow(invT, 3)*cx + 3*pow(invT, 2)*t*p1x + 3*invT*pow(t, 2)*p2x + pow(t, 3)*goal.x;
        double ty = pow(invT, 3)*cy + 3*pow(invT, 2)*t*p1y + 3*invT*pow(t, 2)*p2y + pow(t, 3)*goal.y;

        // 4. Local Transform & Steer Calculation
        double dx = tx - cx;
        double dy = ty - cy;
        double lx =  dx * cos(yaw) + dy * sin(yaw);
        double ly = -dx * sin(yaw) + dy * cos(yaw);

        double L_sq = lx*lx + ly*ly;
        double target_curvature = (2.0 * ly) / L_sq;
        double steer = atan(wheelbase_ * target_curvature);
        steer = clamp(steer, -max_steer_, max_steer_);

        // ── ADAPTIVE SPEED FIX ──
        // Slow down based on how hard we are steering. 
        // 90-degree turns usually require significant steering.
        double target_speed = speed_;
        double steer_ratio = std::abs(steer) / max_steer_;
        
        if (steer_ratio > 0.2) {
            // Gradually reduce speed to 60% of base speed at full steering lock
            double reduction = 1.0 - (steer_ratio * 0.4);
            target_speed = speed_ * std::max(0.6, reduction);
        }

        // 5. Publish
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = this->get_clock()->now();
        drive_msg.header.frame_id = "base_link";
        drive_msg.drive.steering_angle = steer;
        drive_msg.drive.speed = target_speed;
        drive_pub_->publish(drive_msg);

        publish_goal_marker(goal);
    }

    int find_closest_index(double cx, double cy) {
        int closest_idx = last_closest_idx_;
        double min_d = 1e9;
        int n = waypoints_.size();
        for (int k = 0; k < n/2; ++k) {
            int idx = (last_closest_idx_ + k) % n;
            double d = distance(cx, cy, waypoints_[idx].x, waypoints_[idx].y);
            if (d < min_d) { min_d = d; closest_idx = idx; }
        }
        last_closest_idx_ = closest_idx;
        return closest_idx;
    }

    Waypoint find_goal_point(double cx, double cy, int start_idx, int &goal_idx)
    {
        int n = waypoints_.size();
        for (int k = 0; k < n; ++k) {
            int idx = (start_idx + k) % n;
            int nxt = (idx + 1) % n;
            double d0 = distance(cx, cy, waypoints_[idx].x, waypoints_[idx].y);
            double d1 = distance(cx, cy, waypoints_[nxt].x, waypoints_[nxt].y);

            if (d0 < lookahead_dist_ && d1 >= lookahead_dist_) {
                double t_interp = (lookahead_dist_ - d0) / (d1 - d0 + 1e-9);
                goal_idx = idx;
                return {waypoints_[idx].x + t_interp*(waypoints_[nxt].x - waypoints_[idx].x),
                        waypoints_[idx].y + t_interp*(waypoints_[nxt].y - waypoints_[idx].y)};
            }
        }
        goal_idx = start_idx;
        return waypoints_[(start_idx + 1) % n];
    }

    inline double distance(double x1, double y1, double x2, double y2) const {
        return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
    }

    inline double clamp(double val, double lo, double hi) const {
        return max(lo, min(hi, val));
    }

    void publish_goal_marker(const Waypoint &goal) {
        visualization_msgs::msg::Marker m;
        m.header.stamp = this->get_clock()->now();
        m.header.frame_id = "map";
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.pose.position.x = goal.x; m.pose.position.y = goal.y; m.pose.position.z = 0.1;
        m.scale.x = m.scale.y = m.scale.z = 0.2;
        m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0;
        marker_pub_->publish(m);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}