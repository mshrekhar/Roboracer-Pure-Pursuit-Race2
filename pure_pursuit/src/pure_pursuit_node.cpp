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
    double x, y, yaw;
};

class PurePursuit : public rclcpp::Node
{
private:
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

    double min_lookahead_, max_lookahead_, lookahead_gain_;
    double wheelbase_, max_steer_;
    double max_speed_, min_speed_;
    double brake_gain_;
    string waypoint_file_;

    vector<Waypoint> waypoints_;
    int last_closest_idx_{0};

public:
    PurePursuit() : Node("pure_pursuit_node")
    {
        // All tuneable from CLI:  -p max_speed:=6.0  -p min_speed:=1.5  etc.
        this->declare_parameter("min_lookahead", 0.8);
        this->declare_parameter("max_lookahead", 3.0);
        this->declare_parameter("lookahead_gain", 0.3);
        this->declare_parameter("wheelbase", 0.33);
        this->declare_parameter("max_steer_angle", 0.4189);
        this->declare_parameter("max_speed", 5.0);
        this->declare_parameter("min_speed", 1.5);
        this->declare_parameter("brake_gain", 1.0);  // higher = more aggressive braking in turns
        this->declare_parameter("waypoint_file", "waypoints.csv");

        min_lookahead_  = this->get_parameter("min_lookahead").as_double();
        max_lookahead_  = this->get_parameter("max_lookahead").as_double();
        lookahead_gain_ = this->get_parameter("lookahead_gain").as_double();
        wheelbase_      = this->get_parameter("wheelbase").as_double();
        max_steer_      = this->get_parameter("max_steer_angle").as_double();
        max_speed_      = this->get_parameter("max_speed").as_double();
        min_speed_      = this->get_parameter("min_speed").as_double();
        brake_gain_     = this->get_parameter("brake_gain").as_double();
        waypoint_file_  = this->get_parameter("waypoint_file").as_string();

        load_waypoints(waypoint_file_);

        drive_pub_  = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/pure_pursuit/goal_marker", 10);

        // pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        //     "/pf/viz/inferred_pose", 10,
        //     std::bind(&PurePursuit::pose_callback, this, std::placeholders::_1));

        pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
             "/ego_racecar/odom", 10,
             std::bind(&PurePursuit::pose_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(),
            "Pure pursuit started: speed=[%.1f, %.1f] brake_gain=%.2f lookahead=[%.1f, %.1f]",
            min_speed_, max_speed_, brake_gain_, min_lookahead_, max_lookahead_);
    }

    void load_waypoints(const string &filepath)
    {
        ifstream file(filepath);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Could not open: %s", filepath.c_str());
            return;
        }

        string line;
        while (getline(file, line)) {
            // 1. Skip empty lines or comments
            if (line.empty() || line[0] == '#') continue;

            istringstream ss(line);
            string tok;
            Waypoint wp;
            
            try {
                // 2. Read each token and convert to double
                if(!(getline(ss, tok, ','))) continue; 
                wp.x = stod(tok);
                
                if(!(getline(ss, tok, ','))) continue; 
                wp.y = stod(tok);
                
                if(!(getline(ss, tok, ','))) continue; 
                wp.yaw = stod(tok);

                waypoints_.push_back(wp);
            } catch (const std::invalid_argument& e) {
                // This catches headers like "x,y,yaw" and ignores them
                RCLCPP_DEBUG(this->get_logger(), "Skipping non-numeric line or header.");
                continue;
            } catch (const std::out_of_range& e) {
                RCLCPP_ERROR(this->get_logger(), "Value out of range in CSV.");
                continue;
            }
        }
        RCLCPP_INFO(this->get_logger(), "Successfully loaded %zu waypoints", waypoints_.size());
    }
    void pose_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
    {
        if (waypoints_.empty()) return;

        double cx = pose_msg->pose.pose.position.x;
        double cy = pose_msg->pose.pose.position.y;
        double curr_v = pose_msg->twist.twist.linear.x;

        tf2::Quaternion q(
            pose_msg->pose.pose.orientation.x,
            pose_msg->pose.pose.orientation.y,
            pose_msg->pose.pose.orientation.z,
            pose_msg->pose.pose.orientation.w);
        tf2::Matrix3x3 mat(q);
        double r, p, yaw;
        mat.getRPY(r, p, yaw);

        // Dynamic lookahead
        double lookahead = clamp(min_lookahead_ + lookahead_gain_ * fabs(curr_v),
                                 min_lookahead_, max_lookahead_);

        // Find goal point
        int closest_idx = find_closest_index(cx, cy);
        int goal_idx;
        Waypoint goal = find_goal_point(cx, cy, lookahead, closest_idx, goal_idx);

        // Bezier spline interpolation
        int next_idx = (goal_idx + 1) % waypoints_.size();
        double goal_yaw = atan2(waypoints_[next_idx].y - goal.y,
                                waypoints_[next_idx].x - goal.x);
        double dist = distance(cx, cy, goal.x, goal.y);
        double handle_len = dist * 0.5;

        // Rough steer for adaptive t
        double dx_raw = goal.x - cx;
        double dy_raw = goal.y - cy;
        double ly_raw = -dx_raw * sin(yaw) + dy_raw * cos(yaw);
        double rough_steer = atan(wheelbase_ * 2.0 * ly_raw / (lookahead * lookahead));
        double steer_ratio = clamp(fabs(rough_steer) / max_steer_, 0.0, 1.0);
        double t = 0.1 + steer_ratio * 0.3;

        double p1x = cx + cos(yaw) * handle_len;
        double p1y = cy + sin(yaw) * handle_len;
        double p2x = goal.x - cos(goal_yaw) * handle_len;
        double p2y = goal.y - sin(goal_yaw) * handle_len;

        double invT = 1.0 - t;
        double tx = pow(invT,3)*cx + 3*pow(invT,2)*t*p1x + 3*invT*pow(t,2)*p2x + pow(t,3)*goal.x;
        double ty = pow(invT,3)*cy + 3*pow(invT,2)*t*p1y + 3*invT*pow(t,2)*p2y + pow(t,3)*goal.y;

        // Steering
        double dx = tx - cx, dy = ty - cy;
        double lx =  dx * cos(yaw) + dy * sin(yaw);
        double ly = -dx * sin(yaw) + dy * cos(yaw);
        double L_sq = max(lx*lx + ly*ly, 0.001);
        double steer = clamp(atan(wheelbase_ * 2.0 * ly / L_sq), -max_steer_, max_steer_);

        // ── Speed from steering angle ──
        // |steer| = 0         → max_speed  (straight)
        // |steer| = max_steer → min_speed  (tightest turn)
        // Mapping: speed = max - brake_gain * (|steer|/max_steer) * (max - min)
        double steer_frac = fabs(steer) / max_steer_;  // 0..1
        double target_speed = max_speed_ - brake_gain_ * steer_frac * (max_speed_ - min_speed_);
        target_speed = clamp(target_speed, min_speed_, max_speed_);

        // Publish
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

    Waypoint find_goal_point(double cx, double cy, double ld, int start_idx, int &goal_idx)
    {
        int n = waypoints_.size();
        for (int k = 0; k < n; ++k) {
            int idx = (start_idx + k) % n;
            int nxt = (idx + 1) % n;
            double d0 = distance(cx, cy, waypoints_[idx].x, waypoints_[idx].y);
            double d1 = distance(cx, cy, waypoints_[nxt].x, waypoints_[nxt].y);
            if (d0 < ld && d1 >= ld) {
                double t_interp = (ld - d0) / (d1 - d0 + 1e-9);
                goal_idx = idx;
                return {waypoints_[idx].x + t_interp*(waypoints_[nxt].x - waypoints_[idx].x),
                        waypoints_[idx].y + t_interp*(waypoints_[nxt].y - waypoints_[idx].y),
                        waypoints_[idx].yaw};
            }
        }
        goal_idx = (start_idx + 5) % n;
        return waypoints_[goal_idx];
    }

    inline double distance(double x1, double y1, double x2, double y2) const {
        return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
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
        m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0;
        marker_pub_->publish(m);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}