#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    map<vector<BoundingBox>::iterator, cv::Point> mean;
    map<vector<BoundingBox>::iterator, cv::Point> std;

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        

        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
            //cout << pt.x << endl;
            //mean[enclosingBoxes[0]].x += pt.x;
            //mean[enclosingBoxes[0]].y += pt.y;
        }

    } // eof loop over all Lidar points
    // meanx = meanx/enclosingBoxesInter.size();
    // meany = meany/enclosingBoxesInter.size();
    //cout << "debug 0" << endl;

    for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
    {
        if (it2->lidarPoints.size() != 0)
        {
            //myfile << to_string(it2->lidarPoints.size()) + ",";
            cout << "Before filter : " << it2->lidarPoints.size() << endl;
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
            for (auto it3 = it2->lidarPoints.begin(); it3 != it2->lidarPoints.end(); ++it3)
            {
                pcl::PointXYZ point;
                point.x = it3->x;
                point.y = it3->y;
                point.z = it3->z;
                //cout << "debug 1" << endl;
                cloud->push_back(point);
                //cout << "debug 1.2" << endl;
            }
            //cout << "debug 2" << endl;
            //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud (cloud);
            sor.setMeanK (50);
            sor.setStddevMulThresh (1.0);
            sor.filter (*cloud_filtered);
            
            it2->lidarPoints.clear();
            for(auto it3 = cloud_filtered->begin(); it3 != cloud_filtered->end(); ++it3)
            {
                LidarPoint filtered_point;
                filtered_point.x = it3->x;
                filtered_point.y = it3->y;
                filtered_point.z = it3->z;
                it2->lidarPoints.push_back(filtered_point);
            }
        //myfile << to_string(it2->lidarPoints.size()) + ",";
        cout << "After filter : " << it2->lidarPoints.size() << endl;
        }
        
    }
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for(auto it = kptMatches.begin(); it != kptMatches.end(); ++it) 
    {
        auto &currentPoint = kptsCurr[it->trainIdx].pt;
        if (boundingBox.roi.contains(currentPoint)) {
            boundingBox.kptMatches.push_back(*it);
        }
    }

    double mean = 0;

    for (const auto& it : boundingBox.kptMatches) {
        cv::KeyPoint currentPoint = kptsCurr.at(it.trainIdx);
        cv::KeyPoint prevPoint = kptsPrev.at(it.queryIdx);
        double dist = cv::norm(currentPoint.pt - prevPoint.pt);
        mean += dist;
    }

    mean = mean / boundingBox.kptMatches.size();

    double ratio = 1.4;

    for (auto it = boundingBox.kptMatches.begin(); it < boundingBox.kptMatches.end();) {
        cv::KeyPoint currentPoint = kptsCurr.at(it->trainIdx);
        cv::KeyPoint prevPoint = kptsPrev.at(it->queryIdx);
        double dist = cv::norm(currentPoint.pt - prevPoint.pt);

        if (dist >= mean * ratio) {
            boundingBox.kptMatches.erase(it);
        }
        else {
            it++;
        }
    }
    
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    cout << "TTC camera = " << TTC << endl;
    // EOF STUDENT TASK
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
     // auxiliary variables
    double dT = 1/frameRate; // time between two measurements in seconds

    // find closest distance to Lidar points 
    double minXPrev = 1e9, minXCurr = 1e9;
    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it) 
    {
        minXPrev = minXPrev>it->x ? it->x : minXPrev;
    }

    for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it) 
    {
        minXCurr = minXCurr>it->x ? it->x : minXCurr;
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev-minXCurr);
    cout << "TTC lidar = " << TTC << endl;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //cout << prevFrame.boundingBoxes.size() << endl << currFrame.boundingBoxes.size();
    map<int,map<int,int>> check;
    for (auto it1 = matches.begin(); it1 != matches.end(); ++it1)
    {

        cv::Point prev_point, current_point;
        vector<int> prev_matches, curr_matches;
        prev_point.x = (prevFrame.keypoints[it1->queryIdx]).pt.x;
        prev_point.y = (prevFrame.keypoints[it1->queryIdx]).pt.y;
        current_point.x = (currFrame.keypoints[it1->trainIdx]).pt.x;
        current_point.y = (currFrame.keypoints[it1->trainIdx]).pt.y;
        
        //cout << "debug 1.1" << endl;
        for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); ++it2)
        {
            if ((*it2).roi.contains(prev_point))
            {
                prev_matches.push_back((*it2).boxID);
            }
        }
        //cout << "debug 1.2" << endl;
        for (auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); ++it2)
        {
            if ((*it2).roi.contains(current_point))
            {
                curr_matches.push_back((*it2).boxID);
            }
        }
        //cout << "debug 1.3" << endl;
        // if ((prev_matches.size() == 1) && (curr_matches.size() == 1))
        // { 
        //     // add Lidar point to bounding box
        //     check[curr_matches[0]][prev_matches[0]] += 1;
        // }
       //cout << check.size() << endl; 
       for(auto it2 = curr_matches.begin(); it2 != curr_matches.end(); ++it2)
       {
           for (auto it3 = prev_matches.begin(); it3 != prev_matches.end(); ++it3)
           {
               check[*it2][*it3] += 1;
           }
       }
    }

    for (auto it1 = check.begin(); it1 != check.end(); ++it1)
    {
        //cout<<it1->second<< endl;
        //map<int,int> aux = *it1;
        int max_id = 0;
        int max_ocurrence = 0;
        for (auto it2 = check[it1->first].begin(); it2 != check[it1->first].end(); ++it2)
        {
            if (max_ocurrence < it2->second)
            {
                max_ocurrence = it2->second;
                max_id = it2->first;
            }   
            //cout << "(" <<it1->first<<","<<it2->first<<") = "<<it2->second << endl;
        }
        if (max_ocurrence >= 20)
        {
            cout << "Best" << endl;
            cout << "(" <<it1->first<<","<<max_id<<") = "<<max_ocurrence<< endl;
            bbBestMatches[it1->first] = max_id;
            cout << "______" << endl;
        }
    } 
    // ...
}
