#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

using namespace std;

// ──────────────────────────────────────────────
// Simple 2-D waypoint (x, y in map frame)
// ──────────────────────────────────────────────
struct Waypoint {
    double x, y;
};

class PurePursuit : public rclcpp::Node
{
private:
    // ── ROS interfaces ──────────────────────────
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

    // ── Parameters ──────────────────────────────
    double lookahead_dist_;   // L  – lookahead distance (metres)
    double wheelbase_;        // distance between axles (metres)
    double speed_;            // commanded forward speed (m/s)
    double max_steer_;        // steering angle hard limit (radians)
    string waypoint_file_;    // path to CSV  "x,y" per line

    // ── State ────────────────────────────────────
    vector<Waypoint> waypoints_;
    int   last_closest_idx_{0};   // used to keep search monotone

public:
    PurePursuit() : Node("pure_pursuit_node")
    {
        // ── Declare & get parameters ─────────────
        this->declare_parameter("lookahead_dist",  1.5);
        this->declare_parameter("wheelbase",       0.33);   // F1TENTH ~0.33 m
        this->declare_parameter("speed",           1.5);
        this->declare_parameter("max_steer_angle", 0.4189); // ~24 degrees
        this->declare_parameter("waypoint_file",   "waypoints.csv");
        this->declare_parameter("pose_topic",      "/pf/viz/inferred_pose");
        this->declare_parameter("drive_topic",     "/drive");

        lookahead_dist_ = this->get_parameter("lookahead_dist").as_double();
        wheelbase_      = this->get_parameter("wheelbase").as_double();
        speed_          = this->get_parameter("speed").as_double();
        max_steer_      = this->get_parameter("max_steer_angle").as_double();
        waypoint_file_  = this->get_parameter("waypoint_file").as_string();

        string pose_topic  = this->get_parameter("pose_topic").as_string();
        string drive_topic = this->get_parameter("drive_topic").as_string();

        // ── Load waypoints ───────────────────────
        load_waypoints(waypoint_file_);
        RCLCPP_INFO(this->get_logger(),
                    "Loaded %zu waypoints from %s",
                    waypoints_.size(), waypoint_file_.c_str());

        // ── Publishers ───────────────────────────
        drive_pub_  = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
                          drive_topic, 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
                          "/pure_pursuit/goal_marker", 10);

        // ── Subscriber ───────────────────────────
        pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            pose_topic, 10,
            std::bind(&PurePursuit::pose_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Pure Pursuit node started. "
                    "L=%.2f m, v=%.2f m/s", lookahead_dist_, speed_);
    }

    // ────────────────────────────────────────────
    // Load waypoints from a CSV file ("x,y" per row)
    // ────────────────────────────────────────────
    void load_waypoints(const string &filepath)
    {
        ifstream file(filepath);
        if (!file.is_open()) {
            RCLCPP_WARN(this->get_logger(),
                        "Cannot open waypoint file: %s  — "
                        "node will wait. Re-set the 'waypoint_file' parameter "
                        "or pass it on the command line:\n"
                        "  ros2 run pure_pursuit pure_pursuit_node "
                        "--ros-args -p waypoint_file:=/abs/path/to/waypoints.csv",
                        filepath.c_str());
            return;
        }
        string line;
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#') continue; // skip comments
            istringstream ss(line);
            string tok;
            Waypoint wp;
            getline(ss, tok, ','); wp.x = stod(tok);
            getline(ss, tok, ','); wp.y = stod(tok);
            waypoints_.push_back(wp);
        }
    }

    // ────────────────────────────────────────────
    // Main callback – fires on every new pose
    // ────────────────────────────────────────────
    void pose_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
    {
        if (waypoints_.empty()) return;

        // ── 1. Extract current pose in map frame ─
        double cx = pose_msg->pose.pose.position.x;
        double cy = pose_msg->pose.pose.position.y;

        // Yaw from quaternion
        tf2::Quaternion q(
            pose_msg->pose.pose.orientation.x,
            pose_msg->pose.pose.orientation.y,
            pose_msg->pose.pose.orientation.z,
            pose_msg->pose.pose.orientation.w);
        tf2::Matrix3x3 mat(q);
        double roll, pitch, yaw;
        mat.getRPY(roll, pitch, yaw);

        // ── 2. Find path point closest to vehicle ─
        //    (start search from last_closest_idx_ to stay monotone)
        int   closest_idx  = last_closest_idx_;
        double closest_dist = distance(cx, cy,
                                       waypoints_[closest_idx].x,
                                       waypoints_[closest_idx].y);

        int   n = static_cast<int>(waypoints_.size());
        for (int k = 0; k < n; ++k) {
            int   idx  = (last_closest_idx_ + k) % n;
            double d   = distance(cx, cy, waypoints_[idx].x, waypoints_[idx].y);
            if (d < closest_dist) {
                closest_dist = d;
                closest_idx  = idx;
            }
            // Stop searching once we've wrapped more than halfway
            if (k > n / 2) break;
        }
        last_closest_idx_ = closest_idx;

        // ── 3. Find goal point ≈ L ahead ──────────
        //    Walk forward from closest until dist ≥ L,
        //    then interpolate for the exact lookahead point.
        Waypoint goal = find_goal_point(cx, cy, closest_idx);

        // ── 4. Transform goal to vehicle (rear-axle) frame ──
        //    Lecture frame: x forward, y left
        double dx = goal.x - cx;
        double dy = goal.y - cy;
        double gx =  dx * cos(yaw) + dy * sin(yaw);  // longitudinal
        double gy = -dx * sin(yaw) + dy * cos(yaw);  // lateral  (+ = left)

        // ── 5. Compute curvature and steering angle ──
        //    From lecture derivation:
        //       r = L² / (2|y|)   →   γ = 2|y| / L²
        //    Then convert to front-wheel angle via kinematic bicycle model:
        //       δ = arctan( L_wb * γ )   with sign from gy
        double L_actual = sqrt(gx * gx + gy * gy);  // real distance to goal
        if (L_actual < 1e-6) return;                 // degenerate, skip

        // Curvature (signed – positive curves left)
        double curvature = 2.0 * gy / (L_actual * L_actual);

        // Steering angle (kinematic bicycle model)
        double steer = atan(wheelbase_ * curvature);

        // Hard-clamp to physical limits
        steer = clamp(steer, -max_steer_, max_steer_);

        // ── 6. Publish drive command ──────────────
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp    = this->get_clock()->now();
        drive_msg.header.frame_id = "base_link";
        drive_msg.drive.steering_angle = steer;
        drive_msg.drive.speed          = speed_;
        drive_pub_->publish(drive_msg);

        // ── 7. (Optional) visualise the goal point ─
        publish_goal_marker(goal);

        RCLCPP_DEBUG(this->get_logger(),
                     "goal=(%.2f,%.2f) gy=%.3f L=%.3f steer=%.3f°",
                     goal.x, goal.y, gy, L_actual, steer * 180.0 / M_PI);
    }

    // ────────────────────────────────────────────
    // Walk forward from closest_idx until a waypoint
    // at distance ≥ lookahead_dist_ is found, then
    // interpolate to get a point exactly L away.
    // ────────────────────────────────────────────
    Waypoint find_goal_point(double cx, double cy, int start_idx)
    {
        int n = static_cast<int>(waypoints_.size());

        // Forward search
        for (int k = 0; k < n; ++k) {
            int   idx = (start_idx + k) % n;
            int   nxt = (idx + 1) % n;
            double d0  = distance(cx, cy, waypoints_[idx].x, waypoints_[idx].y);
            double d1  = distance(cx, cy, waypoints_[nxt].x, waypoints_[nxt].y);

            // The segment idx→nxt straddles the lookahead circle
            if (d0 < lookahead_dist_ && d1 >= lookahead_dist_) {
                // Interpolate along the segment
                double t = (lookahead_dist_ - d0) / (d1 - d0 + 1e-9);
                t = clamp(t, 0.0, 1.0);
                Waypoint goal;
                goal.x = waypoints_[idx].x + t * (waypoints_[nxt].x - waypoints_[idx].x);
                goal.y = waypoints_[idx].y + t * (waypoints_[nxt].y - waypoints_[idx].y);
                return goal;
            }
        }

        // Fallback: just return the waypoint farthest forward that we found
        // (handles case where all points are inside or outside the circle)
        return waypoints_[(start_idx + 1) % n];
    }

    // ────────────────────────────────────────────
    // Euclidean distance helper
    // ────────────────────────────────────────────
    inline double distance(double x1, double y1, double x2, double y2) const
    {
        double dx = x1 - x2, dy = y1 - y2;
        return sqrt(dx * dx + dy * dy);
    }

    // ────────────────────────────────────────────
    // Clamp helper (std::clamp requires C++17)
    // ────────────────────────────────────────────
    inline double clamp(double val, double lo, double hi) const
    {
        return max(lo, min(hi, val));
    }

    // ────────────────────────────────────────────
    // Publish a sphere marker at the goal point for RViz
    // ────────────────────────────────────────────
    void publish_goal_marker(const Waypoint &goal)
    {
        visualization_msgs::msg::Marker m;
        m.header.stamp    = this->get_clock()->now();
        m.header.frame_id = "map";
        m.ns              = "pure_pursuit";
        m.id              = 0;
        m.type            = visualization_msgs::msg::Marker::SPHERE;
        m.action          = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = goal.x;
        m.pose.position.y = goal.y;
        m.pose.position.z = 0.1;
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.2;
        m.color.r = 1.0; m.color.g = 0.3; m.color.b = 0.0; m.color.a = 1.0;
        marker_pub_->publish(m);
    }

    ~PurePursuit() {}
};

// ──────────────────────────────────────────────
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}