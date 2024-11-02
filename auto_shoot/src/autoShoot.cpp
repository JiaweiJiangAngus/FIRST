//source install/local_setup.bash
//ros2 run auto_shoot auto_shoot_node
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tdt_interface/msg/receive_data.hpp>
#include <tdt_interface/msg/send_data.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ml.hpp>
#include <exception>
#include <string>
#include <map>
#include <vector>
#include <cmath>
using namespace std;
using namespace cv;
using namespace cv::ml;

vector<vector<int>> myColors {{0,75,150,179,255,255},//red
                              {180,255,254,180,255,254},//blue
                              {0,0,130,103,10,255},
                              {180,255,254,180,255,254}};

int hmin = 0, smin = 0, vmin = 0;
int hmax = 180, smax = 255, vmax = 255;
Mat img, imgErode, imgDil, imgcrop, roi;
int mubiao=0;
int turningaspect=1;
double turningupdown=0;
bool isred=0,mapinited=0;
int sumtime=0;
int lastdirect=3;
double map_height,map_width;
Mat usingmap, usingcolormap;
double ax0=0,vx0=0,vx1=0,sx0=0,sx1=0,sx2=0,TTime=0;

vector<Point2f> thepath[6];

const int dire[5][4]={{0,3,1,2},{1,2,0,3},{2,3,1,0},{3,2,0,1},{4,4,4,4}};
int a[51][51];
struct light
{
    Point2f cent;
    Rect r;
};
vector<light>rectcent;
struct Trect
{
    vector<light>l;
    Point2f cent;
    Rect numr;
    Mat ifnum;
};
vector<Trect> num;
class Armor
{
    public:
        Armor(Trect t,int i) : m(t), n(i) {}
        Armor() : m(Trect()), n(0) {}
        virtual ~Armor() {}
        void operator()(const Trect &t, int i,const Mat &rvec = Mat(),const Mat &tvec = Mat(), double dis = 0.0) const {}
    private:
        Trect m;
        int n;
        Mat rvec,tvec;
        double dis;
};
Armor car[10];

class KalmanFilterWrapper {
public:
    KalmanFilterWrapper(int numStates = 6, int numMeasurements = 3, int numControls = 0);
    ~KalmanFilterWrapper() {}
    void predict(double dt);
    void update(const Vec<float, 3> &measurement);
    Vec<float, 6> getStateEstimate() const { return kf.statePost; }
private:
    KalmanFilter kf;
    double friend worldxoy(Trect &t, int x);
};

KalmanFilterWrapper::KalmanFilterWrapper(int numStates, int numMeasurements, int numControls)
{
    kf = KalmanFilter(numStates, numMeasurements, numControls);
    kf.transitionMatrix = Mat::eye(numStates, numStates, CV_32F);

    // 设置位置部分为单位矩阵
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            kf.transitionMatrix.at<float>(i, j) = (i == j ? 1.0f : 0.0f);
        }
    }

    // 设置速度部分
    float dtFactor = 0.1f;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            kf.transitionMatrix.at<float>(i + 3, j) = (i == j ? dtFactor : 0.0f);
        }
    }

    kf.measurementMatrix = Mat::zeros(numMeasurements, numStates, CV_32F);
    kf.measurementMatrix(Rect(0, 0, 3, 3)).setTo(Scalar::all(1.0f)); // Identity for position
    kf.processNoiseCov = Mat::eye(numStates, numStates, CV_32F) * 0.01f;
    kf.measurementNoiseCov = Mat::eye(numMeasurements, numMeasurements, CV_32F) * 0.1f;
    kf.errorCovPost = Mat::eye(numStates, numStates, CV_32F) * 1000.0f;
    kf.statePost = Vec<float, 6>(0.0f);
}

void KalmanFilterWrapper::predict(double dt)
{
    Mat transitionMatrix = kf.transitionMatrix.clone();
    float dtFactor = static_cast<float>(dt);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            transitionMatrix.at<float>(i + 3, j) = (i == j ? dtFactor : 0.0f);
        }
    }
    kf.transitionMatrix = transitionMatrix;
    kf.predict();
}

void KalmanFilterWrapper::update(const Vec<float, 3> &measurement) {
    Mat matMeasurement = Mat::zeros(3, 1, CV_32F);
    matMeasurement.at<float>(0) = measurement(0);
    matMeasurement.at<float>(1) = measurement(1);
    matMeasurement.at<float>(2) = measurement(2);
    kf.correct(matMeasurement);
}
double dist = 0;
vector<Point2f>imgPoints;
Mat cameraMatrix= (Mat_<double>(3, 3)<< 623.5383,    0.0  ,   640.0,
                                          0.0   , 1108.513,   360.0,
                                          0.0   ,    0.0  ,     1.0 );
Mat distCoeffs= (Mat_<double>(1, 5) << 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );
Point2f hitpoint_ori,hitpoint_now;
double world_dis(Point3f p1,Point3f p2)
{
    double x=p2.x-p1.x,y=p2.y-p1.y,z=p2.z-p1.z;
    return sqrt(x*x+y*y+z*z);
}
int sig(int s)
{
    srand(time(0));
    int x=(rand()*s+13);
    if(x%3==0||x%7==0||x%5==0)
        return 1;
    else 
        return -1;
}
Point3f V[11];
double T[11];
double pitchtheta;
bool in_hit=0;
double worldxoy(Trect &t, int x)
{
    std::vector<Point3f> objPoints =
    {
        Point3f(-67.5f,  7.117f,  26.563f),
        Point3f( 67.5f,  7.117f,  26.563f),
        Point3f(-67.5f,    0.0f,     0.0f),
        Point3f( 67.5f,    0.0f,     0.0f),
        Point3f(-67.5f, -7.117f, -26.563f),
        Point3f( 67.5f, -7.117f, -26.563f)
    };

    Mat rvec, tvec;
    bool flags = solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, false, SOLVEPNP_ITERATIVE);

    Mat R;
    Rodrigues(rvec,R);
    Mat woic=-R.t()*tvec;
    Point3f camaraposi(woic.at<double>(0),woic.at<double>(1),woic.at<double>(2));
    pitchtheta=atan(camaraposi.z/sqrt(camaraposi.x*camaraposi.x+camaraposi.y*camaraposi.y))*180.0/3.1415;
    if(flags)
        dist = world_dis(camaraposi,Point3f(0, 0, 0));
    //cout << "Camera Position in World Coordinates " << " : (" << camaraposi.x << ", " << camaraposi.y 
    //     << ", " << camaraposi.z << ")" << endl << "distance : " << dist/1000.0 << "m" << endl;
    srand(time(0));
    KalmanFilterWrapper kf;
    float vx=0,vy=0,vz=0;

    Vec<float,6>initial_state=Vec<float,6>(
        static_cast<float>(0.0),
        static_cast<float>(0.0),
        static_cast<float>(0.0),
        vx,vy,vz);
        
    kf.kf.statePost=initial_state;
    vx=vx0;
    drawFrameAxes(img, cameraMatrix, distCoeffs, rvec, tvec, 200);
    for (double dt = 0.01; dt <= 1; dt += 0.01)
    {
        kf.predict(dt); // 使用卡尔曼滤波器进行预测
        
        // 获取预测位置
        Vec<float, 6> predicted_position = kf.getStateEstimate();
    //    cout << "Predicted Position in World Coordinates after " << dt << " seconds: (" 
    //         << predicted_position(0) << ", " << predicted_position(1) << ", " << predicted_position(2) << ")" << endl;
        Point3f prep(predicted_position(0),predicted_position(1),predicted_position(2));
        // 模拟新的测量数据
        Point3f point_measurement(
            predicted_position(0) + (int)vx0%1000 / 1000.0f,
            predicted_position(1) + static_cast<float>((rand()%1000)) / static_cast<float>(RAND_MAX / 10.0f),
            predicted_position(2) + static_cast<float>((rand()%1000)) / static_cast<float>(RAND_MAX / 10.0f));

        // 将测量数据转换为 Vec<float, 3>
        Vec<float, 3> measurement(point_measurement.x, point_measurement.y, point_measurement.z);

        // 使用新的测量数据更新卡尔曼滤波器
        kf.update(measurement);
        double d=world_dis(prep,camaraposi);
        double v=30;
        if (fabs(v * dt * 1000.0 - d) <= 200)
        {
            vector<Point3f> preobjectPoints;
            preobjectPoints.push_back(prep);
            vector<Point2f> preimagePoints;
            projectPoints(preobjectPoints, rvec, tvec, cameraMatrix, distCoeffs, preimagePoints);

            if (!preimagePoints.empty())
            {
    //            cout << "[After " << dt << " seconds hit: (" 
    //            << predicted_position(0) << ", " << predicted_position(1) << ", " << predicted_position(2) << ")" << endl;
                Point2f pixel_point = preimagePoints[0];
    //            cout << "Pixel coordinates after distortion correction: (" << pixel_point.x << ", " << pixel_point.y << ")]" << endl;
                if (pixel_point.x >= 0 && pixel_point.x < img.cols && pixel_point.y >= 0 && pixel_point.y < img.rows)
                {
                    circle(img, pixel_point, 7, Scalar(255, 0, 255), FILLED);
                    break;
                }
            }
        }
    }
    //car[x](t, x, rvec, tvec, dist);
    if (!flags)
        cout << "Failed to solve PnP!" << endl;
    return dist;
}
double worldxoy2()
{
    std::vector<Point3f> objPoints =
    {
        Point3f(-67.5f,  7.117f,  26.563f),
        Point3f( 67.5f,  7.117f,  26.563f),
        Point3f(-67.5f,    0.0f,     0.0f),
        Point3f( 67.5f,    0.0f,     0.0f),
        Point3f(-67.5f, -7.117f, -26.563f),
        Point3f( 67.5f, -7.117f, -26.563f)
    };

    Mat rvec, tvec;
    bool flags = solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, false, SOLVEPNP_ITERATIVE);

    Mat R;
    Rodrigues(rvec,R);
    Mat woic=-R.t()*tvec;
    Point3f camaraposi(woic.at<double>(0),woic.at<double>(1),woic.at<double>(2));
    pitchtheta=atan(camaraposi.z/sqrt(camaraposi.x*camaraposi.x+camaraposi.y*camaraposi.y))*180.0/3.1415;
    if(flags)
        dist = world_dis(camaraposi,Point3f(0, 0, 0));
    return dist;
}
Point2f att;
float dbp(const Point2f& p1,const Point2f& p2)
{
    float x=p2.x-p1.x,y=p2.y-p1.y;
    return sqrt(x*x+y*y);
}
double lightarea;
char handy;
void getContours(Mat imgD,int x)
{
    vector<vector<Point>>contours;
    vector<Vec4i>hierarchy;
    findContours(imgD,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    //    drawContours(img,contours,-1,Scalar(255,0,255),10);
    Point myPoint(0,0);
    vector<vector<Point>>conPoly(contours.size());
    vector<Rect>boundRect(contours.size());
    // Scalar lower(hmin,smin,vmin);
    // Scalar upper(hmax,smax,vmax);
    // Mat mask;
    // inRange(img,lower,upper,mask);
    // imshow("imgmask",mask);
    for(int i=0;i<contours.size();i++)
    {
        double area=contourArea(contours[i]);
        //cout << area << endl;
        string objectType;
        if(area>10&&x!=2&&x!=3)
        {
            float peri=arcLength(contours[i],true);
            approxPolyDP(contours[i],conPoly[i],0.02*peri,true);
            //drawContours(img, conPoly, i, Scalar(255,0,255), 10);
            boundRect[i]=boundingRect(conPoly[i]);
            if (boundRect[i].height/boundRect[i].width>=2.5&&boundRect[i].height/boundRect[i].width<=7)
            {
                rectangle(img,boundRect[i].tl(),boundRect[i].br(),Scalar(40,255,40),2);
                light tmp;
                tmp.cent=Point2f(boundRect[i].x+boundRect[i].width/2,boundRect[i].y+boundRect[i].height/2);
                tmp.r=boundRect[i];
                rectcent.push_back(tmp);
                mubiao=3;
                att=tmp.cent;
                lightarea=max(lightarea,area);
            }
        }
        if(area>80&&area<29000&&(x==2||x==3))
        {
            float peri=arcLength(contours[i],true);
            approxPolyDP(contours[i],conPoly[i],0.02*peri,true);
            //drawContours(img, conPoly, i, Scalar(255,0,255), 10);
            boundRect[i]=boundingRect(conPoly[i]);
            Point2f cpt(boundRect[i].x+boundRect[i].width/2,boundRect[i].y+boundRect[i].height/2);
            int sum=0;
            vector<light>tmp0;
            for(int j=0;j<rectcent.size();j++)
            {
                if(dbp(rectcent[j].cent,cpt)<=100)
                {
                    sum++;
                    tmp0.push_back(rectcent[j]);
                }
            }
            if(sum>=2)
            {
                if(boundRect[i].width/boundRect[i].height<0.8&&boundRect[i].width/boundRect[i].height>0.2)
                {
                    Trect tmp1;
                    tmp1.cent=cpt;
                    tmp1.ifnum=img(boundRect[i]).clone();
                    while(!tmp0.empty())
                    {
                        tmp1.l.push_back(tmp0.back());
                        tmp0.pop_back();
                    }
                    tmp1.numr=boundRect[i];
                    num.push_back(tmp1);
                }
            }
        }
    }
}

void findColor(Mat img)
{
    Mat imgHSV;
    cvtColor(img,imgHSV,COLOR_BGR2HSV);
    for(int i=0;i<myColors.size()-isred;i++)
    {
        Scalar lower(myColors[i][0],myColors[i][1],myColors[i][2]);
        Scalar upper(myColors[i][3],myColors[i][4],myColors[i][5]);
        Mat mask;
        inRange(imgHSV,lower,upper,mask);
        //imshow(to_string(i), mask);
        getContours(mask,i);
    }
}
Point2f closepoint;
bool cmp(light x,light y)
{
    return dbp(x.cent,closepoint)<dbp(y.cent,closepoint);
}
bool cmp0(Rect x,Rect y)
{
    return x.area()+(double)x.width/x.height*10000>y.area()+(double)x.width/x.height*10000;
}
light re[1001];
// void classifyNumbers()
// {
//     // 加载已经训练好的SVM模型
//     Ptr<SVM>svm=SVM::load("/home/jwj/Desktop/campus_cptt/T-DTtask4/src/auto_shoot/svm_model.xml");
//     // 获取模型的变量数量
//     int varCount=svm->getVarCount();
//     Rect hitfirst[101];
//     int tot0=0;
//     double maxdis=-99999;
//     for(int i=0;i<num.size();i++)
//     {
//         Mat gray;
//         cvtColor(num[i].ifnum,gray,COLOR_BGR2GRAY);
//         equalizeHist(gray,gray);
//         threshold(gray,gray,0,255,THRESH_BINARY|THRESH_OTSU);
//         Mat resized;
//         resize(gray,resized,Size(20,20));
//         // 将图像拉平成一维数组
//         Mat features=resized.reshape(1,1);
//         features.convertTo(features,CV_32F);
//         // 检查特征向量的列数是否与模型中的变量数量匹配
//         if(features.cols!=varCount)
//             continue;
//         float prediction=svm->predict(features);
//         if(prediction)
//         {
//             int predictedNumber=prediction;
//             //rectangle(img,num[i].numr.tl(),num[i].numr.br(),Scalar(40,255,40),2);
//             hitfirst[++tot0]=num[i].numr;
//             mubiao=1;
//             circle(img,num[i].cent,3,Scalar(255,0,30),FILLED);
//             putText(img,to_string(predictedNumber),{num[i].numr.x,num[i].numr.y+num[i].numr.height-5},FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 4, 8);
//             int tot=0;
//             while(!num[i].l.empty())
//             {
//                 re[++tot]=num[i].l.back();
//                 num[i].l.pop_back();
//             }
//             closepoint=num[i].cent;
//             sort(re+1,re+tot+1,cmp);
//             light left_light=re[1],right_light=re[2];
//             if(left_light.r.x>=right_light.r.x)
//                 swap(left_light,right_light);
//             imgPoints.push_back(Point2f(left_light.r.x+left_light.r.width/2.0,left_light.r.y));
//             imgPoints.push_back(Point2f(right_light.r.x+right_light.r.width/2.0,right_light.r.y));
//             imgPoints.push_back(left_light.cent);
//             imgPoints.push_back(right_light.cent);
//             imgPoints.push_back(Point2f(left_light.r.x+left_light.r.width/2.0,left_light.r.y+left_light.r.height));
//             imgPoints.push_back(Point2f(right_light.r.x+right_light.r.width/2.0,right_light.r.y+right_light.r.height));
//             maxdis=max(worldxoy(num[i],predictedNumber),maxdis);
//             imgPoints.clear();
//         }
//     }
//     sort(hitfirst+1,hitfirst+tot0+1,cmp0);
//     rectangle(img,hitfirst[1].tl(),hitfirst[1].br(),Scalar(0,255,255),4);
//     hitpoint_ori=Point2f(hitfirst[1].x+hitfirst[1].width/2,hitfirst[1].y+hitfirst[1].height/2);
//     circle(img,hitpoint_ori,7,Scalar(0,255,255),FILLED);
//     if(mubiao==1)
//     {
//         if(hitpoint_ori.x<540)
//         {
//             turningaspect=-1;
//             mubiao=2;
//         }
//         else if(hitpoint_ori.x>740)
//         {
//             turningaspect=1;
//             mubiao=2;
//         }
//         if(hitpoint_ori.y<330)
//         {
//             turningupdown=1;
//             mubiao=2;
//         }
//         else if(hitpoint_ori.y>390)
//         {
//             turningupdown=-1;
//             mubiao=2;
//         }
//         else
//             turningupdown=0;
//     }
//     //cout<<maxdis<<endl;
//     //cout<<hitfirst[1].area()<<endl;
//     for(int i=2;i<=tot0;i++)
//         rectangle(img,hitfirst[i].tl(),hitfirst[i].br(),Scalar(40,255,40),2);
//     num.clear();
// }
double target_deltax=0,target_deltay=0;
bool fastturn=0;
double deltayaw=0;
void classifyNumbers()
{
    Rect hitfirst[101];
    int tot0=0;
    double maxdis=-99999;

    if(num.size()!=0)
    {
        for(int i=0;i<num.size();i++)
        {
            hitfirst[++tot0]=num[i].numr;
            mubiao=1;
            circle(img,num[i].cent,3,Scalar(255,0,30),FILLED);
            int tot=0;
            while(!num[i].l.empty())
            {
                re[++tot]=num[i].l.back();
                num[i].l.pop_back();
            }
            closepoint=num[i].cent;
            sort(re+1,re+tot+1,cmp);
            light left_light=re[1],right_light=re[2];
            if(left_light.r.x>=right_light.r.x)
                swap(left_light,right_light);
            imgPoints.push_back(Point2f(left_light.r.x+left_light.r.width/2.0,left_light.r.y));
            imgPoints.push_back(Point2f(right_light.r.x+right_light.r.width/2.0,right_light.r.y));
            imgPoints.push_back(left_light.cent);
            imgPoints.push_back(right_light.cent);
            imgPoints.push_back(Point2f(left_light.r.x+left_light.r.width/2.0,left_light.r.y+left_light.r.height));
            imgPoints.push_back(Point2f(right_light.r.x+right_light.r.width/2.0,right_light.r.y+right_light.r.height));
            maxdis=max(worldxoy(num[i],4),maxdis);
            imgPoints.clear();
        }
    }
    // else
    // {
    //     double mindist=999999,min1=0,min2=0;
    //     for(int i=0;i<rectcent.size();i++)
    //     {
    //         for(int j=i+1;j<rectcent.size();j++)
    //         {
    //             if(mindist>dbp(rectcent[i].cent,rectcent[j].cent))
    //             {
    //                 mindist=dbp(rectcent[i].cent,rectcent[j].cent);
    //                 min1=i;
    //                 min2=j;
    //             }
    //         }
    //     }
    //     if(!(min1==0&&min2==0)&&mindist<400)
    //     {
    //         light left_light=rectcent[min1],right_light=rectcent[min2];
    //         if(left_light.r.x>=right_light.r.x)
    //             swap(left_light,right_light);
    //         imgPoints.push_back(Point2f(left_light.r.x+left_light.r.width/2.0,left_light.r.y));
    //         imgPoints.push_back(Point2f(right_light.r.x+right_light.r.width/2.0,right_light.r.y));
    //         imgPoints.push_back(left_light.cent);
    //         imgPoints.push_back(right_light.cent);
    //         imgPoints.push_back(Point2f(left_light.r.x+left_light.r.width/2.0,left_light.r.y+left_light.r.height));
    //         imgPoints.push_back(Point2f(right_light.r.x+right_light.r.width/2.0,right_light.r.y+right_light.r.height));
    //         worldxoy2();
    //         imgPoints.clear();
    //         if(dist<800)
    //             dist=800;
    //         if(dist>2500)
    //             dist=2500;
    //     }
    //     else
    //         dist=0;
    // }
    sort(hitfirst+1,hitfirst+tot0+1,cmp0);
    rectangle(img,hitfirst[1].tl(),hitfirst[1].br(),Scalar(0,255,255),4);
    hitpoint_ori=Point2f(hitfirst[1].x+hitfirst[1].width/2,hitfirst[1].y+hitfirst[1].height/2);
    stringstream s1;
    s1 << "hitpoint: (" << hitpoint_ori.x << " , " << hitpoint_ori.y << " ) ";
    string hit_text = s1.str();
    putText(img, hit_text, cv::Point(img.rows - 200, img.rows - 80), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 2);
    if(mubiao==3)
    {
        hitpoint_ori=att;
        mubiao=1;
    }
    turningupdown=0;
    if(mubiao==1)
    {
        fastturn=0;
        if(hitpoint_ori.x<630)
        {
            turningaspect=-1;
            mubiao=2;
            if(hitpoint_ori.x>430)
                fastturn=1;
        }
        else if(hitpoint_ori.x>650)
        {
            turningaspect=1;
            mubiao=2;
            if(hitpoint_ori.x>850)
                fastturn=1;
        }
        sx2=(sx1==0?(hitpoint_ori.x):sx1)-deltayaw;
        sx1=(sx0==0?(hitpoint_ori.x):sx0);
        sx0=hitpoint_ori.x;
        vx1=(vx0==0?(sx0-sx1)/TTime:vx0);
        vx0=(sx0-sx1)/TTime;
        ax0=(vx0-vx1)/TTime/2.0;
        hitpoint_ori.x=sx0+vx0*TTime+ax0*TTime*TTime/2.0;
        if(hitpoint_ori.x>1200)
            hitpoint_ori.x=1200;
        else if(hitpoint_ori.x<80)
            hitpoint_ori.x=80;
        circle(img,hitpoint_ori,7,Scalar(0,255,255),FILLED);
        // if(hitpoint_ori.y<356)
        // {
        //     turningupdown=1;
        //     target_deltay=fabs(720.0-(double)hitpoint_ori.y);
        //     //turningupdown=atan((360.0-(double)hitpoint_ori.y)/720.0)*180/3.1415;
        //     mubiao=2;
        // }
        // else if(hitpoint_ori.y>364)
        // {
        //     turningupdown=-1;
        //     target_deltay=fabs(360.0-(double)hitpoint_ori.y)/360.0*6;
        //     //turningupdown=-atan(((double)hitpoint_ori.y-360.0)/720.0)*180/3.1415;
        //     mubiao=2;
        // }
    }
    //cout<<maxdis<<endl;
    //cout<<hitfirst[1].area()<<endl;
    for(int i=2;i<=tot0;i++)
        rectangle(img,hitfirst[i].tl(),hitfirst[i].br(),Scalar(40,255,40),2);
    num.clear();
}

int b[51][51];
int b0[6][51][51][2];
const int dx[4] = {-1, 0, 0, 1};
const int dy[4] = {0, -1, 1, 0};
int c[6][2],d[6][6];
bool havego[6];
void bfs(int x,int y,int tot,int begin)
{
    if(a[x][y]==0)
    {
        for(int i=1;i<=5;i++)
        {
            if(begin==i)
                continue;
            if(x==c[i][0]&&y==c[i][1]&&d[begin][i]==0)
            {
                d[begin][i]=b[x][y];
                d[i][begin]=b[x][y];
                tot++;
                break;
            }
            else if(x==c[i][0]&&y==c[i][1]&&d[begin][i]>b[x][y])
            {
                d[begin][i]=b[x][y];
                d[i][begin]=b[x][y];
                break;
            }
        }
    }
    if(tot==5)
        return;
    for(int i=0;i<4;i++)
    {
        if(x+dx[i]>=0&&x+dx[i]<=50&&!(c[begin][0]==x+dx[i]&&c[begin][1]==y+dy[i])&&y+dy[i]>=0&&y+dy[i]<=50&&a[x+dx[i]][y+dy[i]]!=-1&&(b[x+dx[i]][y+dy[i]]==0||b[x+dx[i]][y+dy[i]]>b[x][y]+1))
        {
            b[x+dx[i]][y+dy[i]]=b[x][y]+1;
            b0[begin][x+dx[i]][y+dy[i]][0]=x;
            b0[begin][x+dx[i]][y+dy[i]][1]=y;
            bfs(x+dx[i],y+dy[i],tot,begin);
            if(tot==5)
                return;
        }
    }
}
int mintartime[6];
void querenlujing()
{
    int minx=9999999;
    mintartime[0]=0;
    for(int i=1;i<=5;i++)
    {
        for(int j=1;j<=5;j++)
        {
            if(j==i)
                continue;
            for(int k=1;k<=5;k++)
            {
                if(k==i||k==j)
                    continue;
                for(int l=1;l<=5;l++)
                {
                    if(l==i||l==j||l==k)
                        continue;
                    for(int m=1;m<=5;m++)
                    {
                        if(m==i||m==j||m==k||m==l)
                            continue;
                        int sum=d[0][i]+d[i][j]+d[j][k]+d[k][l]+d[l][m];
                        if(sum<minx)
                        {
                            minx=sum;
                            mintartime[1]=i;
                            mintartime[2]=j;
                            mintartime[3]=k;
                            mintartime[4]=l;
                            mintartime[5]=m;
                        }
                    }
                }
            }
        }
    }
}

int checkPixelValue(cv::Mat& map, cv::Point2f pos, int offsetX, int offsetY)
{
    // 确保位置不越界
    int x = static_cast<int>(pos.x + offsetX);
    int y = static_cast<int>(pos.y + offsetY);
    if (x >= 0 && y >= 0 && x < map.cols && y < map.rows)
    {
        unsigned char pixelValue = map.at<unsigned char>(y, x);
        return (pixelValue == 0) ? 1 : 0;
    }
    return 1; // 越界时，默认为墙壁
}

vector<Point2f> targetcenter;

vector<Rect> targetrect;

void gettarget(Mat imgD,Mat img)
{
    vector<vector<Point>>contours;
    vector<Vec4i> hierarchy;

    findContours(imgD,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    //drawContours(imgD,contours,-1,Scalar(255,0,255),10);

    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());
    for(int i=0;i<contours.size();i++)
    {
        double area=contourArea(contours[i]);
        string objectType;
        if(area>=1000)
        {
            float peri = arcLength(contours[i],true);
            approxPolyDP(contours[i],conPoly[i],0.02*peri,true);
            //drawContours(img,conPoly,i,Scalar(255,0,255),10);
            //cout<<conPoly[i].size()<<endl;
            
            int objcor=(int)conPoly[i].size();

            if(objcor==4)
            {
                float aspRatio =(float) boundRect[i].width/(float) boundRect[i].height;
                boundRect[i]=boundingRect(conPoly[i]);
                rectangle(img,boundRect[i].tl(),boundRect[i].br(),Scalar(0,255,0),5);
                targetrect.push_back(boundRect[i]);
                Point2f cent=Point2f(boundRect[i].x+boundRect[i].width/2,boundRect[i].y+boundRect[i].height/2);
                //circle(img,cent,3,Scalar(0,0,255),FILLED);
                targetcenter.push_back(cent);
            }
        }
    }
}
int I[6],I0;
int direction;
Rect inrect[6];
void fixroad()
{
    for(int i=2;i<=5;i++)
    {
        int j=thepath[i].size()-1;
        int k=2;
        while(thepath[i][j].x!=thepath[i-1][k].x||thepath[i][j].y!=thepath[i-1][k].y)
        {
            thepath[i][j].x=thepath[i-1][k].x;
            thepath[i][j].y=thepath[i-1][k].y;
            j--;
            k++;
        }
    }
}
void initmap()
{
    // namedWindow("Trackbars",(640,200));
    // createTrackbar("Hue Min","Trackbars",&hmin,179);
    // createTrackbar("Hue Max","Trackbars",&hmax,179);
    // createTrackbar("Sat Min","Trackbars",&smin,255);
    // createTrackbar("Sat Max","Trackbars",&smax,255);
    // createTrackbar("Val Min","Trackbars",&vmin,255);
    // createTrackbar("Val Max","Trackbars",&vmax,255);
    erode(usingmap, usingmap, getStructuringElement(MORPH_RECT ,Size(41,41)));
    usingcolormap=usingmap.clone();
    Mat imggray, imgBlur, imgCanny, imgDil = usingmap.clone(), imgErode;
    cvtColor(usingcolormap, usingcolormap, COLOR_RGB2BGR);
    GaussianBlur(imgDil, imgBlur,Size(3,3),5.0);
    Mat kernel = getStructuringElement(MORPH_RECT ,Size(7,7));
    erode(imgBlur, imgErode, getStructuringElement(MORPH_RECT ,Size(11,11)));
    //imshow("ImageE",imgErode);
    gettarget(imgErode,usingcolormap);
    for(int i=0;i<=50;i++)
    {
        for(int j=0;j<=50;j++)
        {
            if((i%2)&&(j%2))
            {
//                circle(usingcolormap,Point2f{i*25.0f,j*25.0f},3,Scalar(255,0,0),FILLED);
                a[i][j]=1;
            }
            else if(!checkPixelValue(usingmap, Point2f{i*25.0f,j*25.0f}, 0, 0))
            {
//                circle(usingcolormap,Point2f{i*25.0f,j*25.0f},3,Scalar(255,0,0),FILLED);
                a[i][j]=1;
            }
            else
            {
//                circle(usingcolormap,Point2f{i*25.0f,j*25.0f},3,Scalar(255,255,0),FILLED);
                a[i][j]=-1;
            }
        }
    }
    for(int i=0;i<targetcenter.size();i++)
    {
        int x=(int)floor(targetcenter[i].x+0.5)/25;
        int y=(int)floor(targetcenter[i].y+0.5)/25;
        a[x][y]=0;
        c[i+1][0]=x;
        c[i+1][1]=y;
        stringstream ss;
        ss << i+1 ;
        string num_text = ss.str();
        putText(usingcolormap, num_text, Point2f{x*25.0f-20,y*25.0f-40}, FONT_HERSHEY_SIMPLEX, 5, Scalar(0, 0, 255), 2);
//        circle(usingcolormap,Point2f{x*25.0f,y*25.0f},3,Scalar(0,255,0),FILLED);
    }
    memset(b,0,sizeof(b));
    bfs(1,1,0,0);
    c[0][0]=1;
    c[0][1]=1;
    b0[0][c[0][0]][c[0][1]][0]=0;
    b0[0][c[0][0]][c[0][1]][1]=0;
    for(int i=1;i<=5;i++)
    {
        memset(b,0,sizeof(b));
        bfs(c[i][0],c[i][1],0,i);
        b0[i][c[i][0]][c[i][1]][0]=0;
        b0[i][c[i][0]][c[i][1]][1]=0;
    }
    querenlujing();
    //cout<<"jwjsb"<<endl;
    for(int i=5;i>=1;i--)
    {
        thepath[i].clear();
        thepath[i].push_back(Point2f{0.0f,0.0f});
        int tmpx=c[mintartime[i]][0];
        int tmpy=c[mintartime[i]][1];
        while(b0[mintartime[i-1]][tmpx][tmpy][0]&&b0[mintartime[i-1]][tmpx][tmpy][1])
        {
            thepath[i].push_back(Point2f{(float)tmpx,(float)tmpy});
            line(usingcolormap,Point2f{(float)tmpx*25.0f,(float)tmpy*25.0f},Point2f{(float)b0[mintartime[i-1]][tmpx][tmpy][0]*25.0f,(float)b0[mintartime[i-1]][tmpx][tmpy][1]*25.0f},Scalar(0,255,255),5);
            int sx=b0[mintartime[i-1]][tmpx][tmpy][0];
            int sy=b0[mintartime[i-1]][tmpx][tmpy][1];
            tmpx=sx;
            tmpy=sy;
        }
        I[i]=thepath[i].size()-1;
        inrect[i]=targetrect[mintartime[i]-1];
        cout<<"mintartime["<<i<<"]="<<mintartime[i]<<endl;
    }
    fixroad();
    I0=1;
    if(thepath[1][I[1]].x==2)
        direction=3;
    else
        direction=2;
    cout<<"JwJSB1"<<endl;
    imshow("The color map",usingcolormap);
    cout<<"JwJSB2"<<endl;
    // findways(0,2,Point2f(25,25),Point2f(25,25));
    // for(int i=0;i<pathnode.size();i++)
    // {
    //     circle(usingmap,pathnode[i],3,Scalar(0,0,0),1);
    // }
    //imshow("The map",usingmap);
    //cvtColor(usingmap, usingmap, COLOR_BGR2GRAY);
    //threshold(usingmap, usingmap, 127, 255, cv::THRESH_BINARY);
}

void showmap(Point2f p)
{
    cout<<turningupdown<<endl;
    circle(usingcolormap,p,3,Scalar(255,0,255),FILLED);
    imshow("The color map",usingcolormap);
    circle(usingcolormap,p,3,Scalar(255,255,255),FILLED);
    // for(int i=1;i<=5;i++)
    // {
    //     cout<<"mintartime["<<i<<"]="<<mintartime[i]<<endl;
    // }
    // cout<<targetcenter.size()<<endl;
    // for(int i=0;i<=5;i++)
    // {
    //     for(int j=0;j<=5;j++)
    //     {
    //         printf("%d ",d[i][j]);
    //     }
    //     printf("\n");
    // }
    //cout<<usingmap;
    //waitKey(0);
}
int nowdir,jusdir;
double rectarea=0;
bool Pointsinrect(Point2f p,Rect r)
{
    if(p.x>r.x&&p.y>r.y&&p.x<r.x+r.width&&p.y<r.y+r.height)
    {
        rectarea=r.width*r.height;
        return 1;
    }
    return 0;
}
int time1=0,state=0,finding=0;
double speed_up=0;
int findtime=0;
int hiterror=0;
double xiuzheng_x=0;
double xiuzheng_y=0;
double realyaw=0;
bool ifreturn_center=0;
int touring(Point2f p,Mat img)
{
    if((finding>0&&state==0)||I[I0]<2)
    {
        findColor(img);
        classifyNumbers();
        rectcent.clear();
        for(int i=0;i<10;i++)
        {
            car[i]=Armor(Trect(),i);
        }
        //cout<<"lightarea:"<<lightarea<<endl;
        if(lightarea>40||I[I0]<2)
        {
            findtime++;
            lightarea=0;
        }
        if(findtime>=2||I[I0]<1)
        {
            state=1;
            findtime=0;
            finding=0;
        }
    }
    if(state==0)
    {
        xiuzheng_x=0;
        xiuzheng_y=0;
        if(Pointsinrect(p,inrect[I0])&&finding==0)
        {
            time1=0;
            in_hit=1;
            finding=1;
        }
        if(direction==3&&p.y>=thepath[I0][I[I0]].y*25+11)
        {
            thepath[I0].pop_back();
            I[I0]--;
        }
        if(direction==2&&p.x>=thepath[I0][I[I0]].x*25+11)
        {
            thepath[I0].pop_back();
            I[I0]--;
        }
        if(direction==1&&p.y<=thepath[I0][I[I0]].y*25-11)
        {
            thepath[I0].pop_back();
            I[I0]--;
        }
        if(direction==0&&p.x<=thepath[I0][I[I0]].x*25-11)
        {
            thepath[I0].pop_back();
            I[I0]--;
        }
        if(thepath[I0][I[I0]].y==thepath[I0][I[I0]-1].y)
        {
            if(p.x-(thepath[I0][I[I0]].x*25)>10)
                xiuzheng_x=-0.1;
            if(p.x-(thepath[I0][I[I0]].x*25)<-10)
                xiuzheng_x=0.1;
            if(thepath[I0][I[I0]].x>thepath[I0][I[I0]-1].x)
                direction=0;
            else
                direction=2;
        }
        if(thepath[I0][I[I0]].x==thepath[I0][I[I0]-1].x)
        {
            if(p.y-(thepath[I0][I[I0]].y*25)>10)
                xiuzheng_y=-0.1;
            if(p.y-(thepath[I0][I[I0]].y*25)<-10)
                xiuzheng_y=0.1;
            if(thepath[I0][I[I0]].y>thepath[I0][I[I0]-1].y)
                direction=1;
            else
                direction=3;
        }
        if(I[I0]>5)
        {
            if(direction&1)
                for(int i=I[I0],j=1;i>=3;i--,j++)
                {
                    if(thepath[I0][i].x!=thepath[I0][i-1].x)
                    {
                        if(j<=3)
                            speed_up=0.1;
                        else
                            speed_up=j-2;
                        break;
                    }
                }
            else
                for(int i=I[I0],j=1;i>=3;i--,j++)
                {
                    if(thepath[I0][i].y!=thepath[I0][i-1].y)
                    {
                        if(j<=3)
                            speed_up=0.3;
                        else
                            speed_up=j-2;
                        break;
                    }
                }
        }
        else
            speed_up=0;
        speed_up=speed_up*(1+0.3*((speed_up)>1?0:speed_up-1));
        return 0;
    }
    else if(state==1)
    {
        rectarea=0;
        finding=0;
        sx2=0,sx1=0,sx0=0;
        vx1=0,vx0=0;
        ax0=0;
        cout<<time1<<endl;
        in_hit=1;
        if(time1>=60)
        {
            if(I0==5)
            {
                mubiao=0;
                time1=0;
                finding=1;
            }
            if(ifreturn_center==1&&I0<5)
            {
                int pixx=(int)((p.x/25.0)+0.5);
                int pixy=(int)((p.y/25.0)+0.5);
                int tmpx=c[mintartime[I0]][0];
                int tmpy=c[mintartime[I0]][1];
                if(pixx<tmpx)
                    for(int i=pixx;i<=tmpx;i++)
                    {
                        thepath[I0+1].push_back(Point2f{(float)i,(float)pixy});
                        I[I0+1]++;
                    }
                else
                    for(int i=pixx;i>=tmpx;i--)
                    {
                        thepath[I0+1].push_back(Point2f{(float)i,(float)pixy});
                        I[I0+1]++;
                    }
                if(pixy<tmpy)
                    for(int i=pixy+1;i<=tmpy;i++)
                    {
                        thepath[I0+1].push_back(Point2f{(float)tmpx,(float)i});
                        I[I0+1]++;
                    }
                else
                    for(int i=pixy;i>tmpy;i--)
                    {
                        thepath[I0+1].push_back(Point2f{(float)tmpx,(float)i});
                        I[I0+1]++;
                    }
                ifreturn_center=0;
            }
            state=0;
            if(I0<5)
                I0++;
            in_hit=0;
        }
        return 0;
    }
    return 0;
}
long long frame_count = 0;
double startt = static_cast<double>(cv::getTickCount());
double fps = 0.0;
int emergency=0;
class AutoShoot : public rclcpp::Node
{
public:
    AutoShoot()
        : Node("auto_shoot_node")
    {
        // 订阅相机图像
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera_image", 10,
            std::bind(&AutoShoot::imageCallback, this, std::placeholders::_1));

        // 订阅云台角度
        receive_data_sub_ = this->create_subscription<tdt_interface::msg::ReceiveData>(
            "/real_angles", 10,
            std::bind(&AutoShoot::receiveCallback, this, std::placeholders::_1));

        // 订阅栅格地图
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 10,
            std::bind(&AutoShoot::mapCallback, this, std::placeholders::_1));

        // 订阅当前机器人位姿
        position_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/position", 10,
            std::bind(&AutoShoot::positionCallback, this, std::placeholders::_1));

        // 订阅当前真实速度
        real_speed_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
            "/real_speed", 10,
            std::bind(&AutoShoot::realSpeedCallback, this, std::placeholders::_1));

        // 订阅目标点位姿
        goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10,
            std::bind(&AutoShoot::goalPoseCallback, this, std::placeholders::_1));

        // 发布目标云台角度
        send_data_pub_ = this->create_publisher<tdt_interface::msg::SendData>("/target_angles", 10);

        // 发布目标速度
        speed_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/target_speed", 10);

        // 发布比赛开始信号
        game_start_pub_ = this->create_publisher<std_msgs::msg::Bool>("/game_start", 10);

        publishGameStartSignal();
        waitKey(20);
        publishGameStartSignal();
        waitKey(20);
        publishGameStartSignal();
    }

private:
    geometry_msgs::msg::PoseStamped::SharedPtr robot_pose= nullptr;
    float real_linear_speed_x , real_linear_speed_y;
    Point2f cur_pos;

    void publishGameStartSignal()
    {
        auto msg = std::make_shared<std_msgs::msg::Bool>();
        msg->data = true;
        game_start_pub_->publish(*msg);
        RCLCPP_INFO(this->get_logger(), "Game start");
    }


    void positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
       /***********************处理自身位置信息**************************/
        robot_pose = msg;
        RCLCPP_INFO(this->get_logger(), "Robot position: [x: %f, y: %f, z: %f]", msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if(mapinited==0)
        {
            initmap();
            mapinited=1;
        }
        cv::Mat frame;
        std::vector<uint8_t> jpeg_data(msg->data.begin(), msg->data.end());
        frame = cv::imdecode(jpeg_data, cv::IMREAD_COLOR);
        frame_count++;
        double current_time = static_cast<double>(cv::getTickCount());
        if (frame_count == 1) {
            startt = current_time;
        }
        double elapsed_time = (current_time - startt) / cv::getTickFrequency();
        if (elapsed_time > 0) {
            fps = frame_count / elapsed_time;
        }
        // 绘制帧率信息
        std::stringstream ss,sss;
        ss << "FPS: " << static_cast<double>(fps);
        std::string fps_text = ss.str();
        if (robot_pose != nullptr)
        {
            // 绘制机器人位置信息
            std::stringstream ss;
            ss << "Robot Position: [" 
            << "x: " << robot_pose->pose.position.x << ", "
            << "y: " << robot_pose->pose.position.y << ", "
            << "z: " << robot_pose->pose.position.z << "]";
            std::string pos_text = ss.str();
            cv::putText(frame, pos_text, cv::Point(10, frame.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }
        fps+=0.000000001;
        TTime=(1.0/fps+0.00001);
        int img_width = frame.cols;
        int img_height = frame.rows;
        sss << "imgsize( " << frame.cols << " , " << frame.rows << " )";
        std::string imgsize_text = sss.str();
        cv::putText(frame, imgsize_text, cv::Point(10, frame.rows - 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, fps_text, cv::Point(frame.cols - 150, frame.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        
        if(frame_count>=1e10||elapsed_time>=1e10)
        {
            frame_count=0;
            elapsed_time=0;
        }
        if (frame.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to decode image.");
            return;
        }
        /********************处理你的图像*********************/
        handy=waitKey(1);
        if(state==1||finding==1)
        {
            img=frame;
            findColor(img);
            classifyNumbers();
            rectcent.clear();
            for(int i=0;i<10;i++)
            {
                car[i]=Armor(Trect(),i);
            }
            if(lightarea>10&&lightarea<50)
            {
                finding=1;
                state=0;
                findtime=0;
            }
            frame=img;
            if(mubiao==0)
            {
                //turningaspect=1*(((int)dist&1)?-1:1);
                yaw+=2*turningaspect;
                if(state==1&&in_hit==1)
                    time1++;
            }
            else if(mubiao==1)
            {
                turningaspect=turningaspect*(-1);
                time1=0;
                if(lightarea<30)
                    pitch+=(352.0-((hitpoint_ori.y)==0?352.0:hitpoint_ori.y))*60.0/720.0/5.0;
                else
                    pitch+=(354.0-((hitpoint_ori.y)==0?356.0:hitpoint_ori.y))*60.0/720.0/2.5;
                finding=2;
                deltayaw=0;
            }
            else if(mubiao==2)
            {
                double tmp=yaw;
                if(hitpoint_ori.x<580.0||hitpoint_ori.x>700.0)
                    yaw+=fabs(640.0-((hitpoint_ori.x)==0?640.0:hitpoint_ori.x-5))*90.0/1280.0/1.2*turningaspect;
                else   
                    yaw+=fabs(640.0-((hitpoint_ori.x)==0?640.0:hitpoint_ori.x-5))*90.0/1280.0/10.0*turningaspect;
                deltayaw=(tmp-yaw)*turningaspect*(-1);
                time1=0;
                pitch+=(358.0-((hitpoint_ori.y)==0?358.0:hitpoint_ori.y-5))*60.0/720.0/5.0;
                finding=2;
                fastturn=0;
            }
            else
                finding=2;
            if(pitch>-5)
                pitch=-5;
            if(pitch<-30)
                pitch=-30;
            if(handy=='a'||handy=='A')
            {
                yaw-=15;
                handy=0;
            }
            else if(handy=='d'||handy=='D')
            {
                yaw+=15;
                handy=0;
            }
            else if(handy=='s'||handy=='S')
            {
                finding=1;
                time1=0;
                handy=0;
            }
            else if(handy=='g'||handy=='G')
            {
                state=0;
                handy=0;
                rectarea=0;
                finding=0;
                sx2=0,sx1=0,sx0=0;
                vx1=0,vx0=0;
                ax0=0;
                if(I0<5)
                    I0++;
                in_hit=0;
            }
            else if(handy=='w'||handy=='W')
            {
                emergency=2;
            }
            cout<<"hitpoint:   "<<hitpoint_ori<<endl;
        }
        cur_pos.x=robot_pose->pose.position.x*25;
        cur_pos.y=robot_pose->pose.position.y*25;
        // // std::stringstream mappos_s;
        // // mappos_s << "imgsize( " << img_width << " , " << img_height << " )";
        // // std::string mappos_s_text = mappos_s.str();
        // //cv::putText(map_image, mappos_s_text, cv::Point(cur_pos), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        
        showmap(cur_pos);
        touring(cur_pos,frame);
        if(handy=='h'||handy=='H')
        {
            if(direction==3||direction==1)
            {
                if(cur_pos.x>thepath[I0][I[I0]].x*25)
                    direction=0;
                else
                    direction=2;
            }
            else if(direction==0||direction==2)
            {
                if(cur_pos.y>thepath[I0][I[I0]].y*25)
                    direction=1;
                else
                    direction=3;
            }
            speed_up=0.5;
        }
        realyaw=yaw;
        while(realyaw>=360)
            realyaw-=360;
        while(realyaw<0)
            realyaw+=360;
        cout<<"lightarea:"<<lightarea<<endl;
        if(dist<=1000&&dist>1&&(state==1||finding==1))
        {
            emergency=1;
            ifreturn_center=1;
        }
        if(dist>3000&&dist>1&&(state==1||finding==1))
        {
            emergency=2;
            ifreturn_center=1;
        }
        if(lightarea<15&&(state==1||finding==1))
        {
            emergency=2;
            //ifreturn_center=1;
        }
        if(lightarea>450&&(state==1||finding==1))
        {
            emergency=1;
            //ifreturn_center=1;
        }
        if(state==0)
        {
            ifreturn_center=0;
            if(direction==0)yaw=90;
            else if(direction==1)yaw=0;
            else if(direction==2)yaw=270;
            else if(direction==3)yaw=180;
            pitch=0;
        }
        if(handy=='f'||handy=='F')
            ifreturn_center=0;
        //yaw=expyaw(yaw,direction);
        stringstream sd;
        sd << "Robot Direction: " <<direction;
        string direc_text = sd.str();
        cv::putText(frame, direc_text, cv::Point(10, frame.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        stringstream ssd;
        ssd << "Next position: ( " <<thepath[I0][I[I0]].x<<" , "<<thepath[I0][I[I0]].y<<" )";
        string nextpos_text = ssd.str();
        cv::putText(frame, nextpos_text, cv::Point(frame.cols - 250, frame.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        
        stringstream s0;
        s0 << "Distance: " << dist <<" finding: " << finding ;
        string dis_text = s0.str();
        putText(frame, dis_text, cv::Point(10, frame.rows - 80), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 2);
        



        rectangle(frame,Point(620,340),Point(660,380),Scalar(255,0,255),2);
        
        imshow("Camera Image", frame);
        cv::waitKey(1);






        
        /******************************************/
        
        auto send_data_msg = std::make_shared<tdt_interface::msg::SendData>();
        send_data_msg->pitch = pitch;
        send_data_msg->yaw = yaw;
        if(handy!='k'&&handy!='K')
            send_data_msg->if_shoot =  state&&mubiao;
        else 
        {
            send_data_msg->if_shoot =  true;
            handy=0;
        }
        send_data_pub_->publish(*send_data_msg);
        mubiao=0;
    }

    void realSpeedCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
    {
        real_linear_speed_x = msg->twist.linear.x;
        real_linear_speed_y = msg->twist.linear.y;
        /****************处理回调速度************************/
        RCLCPP_DEBUG( this->get_logger(), "Real linear speed: [x: %f, y: %f]", real_linear_speed_x, real_linear_speed_y);

        double expectspeed_x=0,expectspeed_y=0;
        if(emergency==1)
        {
            if(fabs(sin(realyaw/180.0*3.1415926))>fabs(cos(realyaw/180.0*3.1415926)))
                expectspeed_x=sin(realyaw/180.0*3.1415926)*1.5;
            else
                expectspeed_y=cos(realyaw/180.0*3.1415926)*1.5;
            finding=1;
            state=0;
            findtime=0;
            emergency=0;
        }
        else if(emergency==2)
        {
            if(fabs(sin(realyaw/180.0*3.1415926))>fabs(cos(realyaw/180.0*3.1415926)))
                expectspeed_x=-sin(realyaw/180.0*3.1415926);
            else
                expectspeed_y=-cos(realyaw/180.0*3.1415926);
            emergency=0;
        }
        else
        {
            if(state==0)
            {
                if(direction==0)
                {
                    expectspeed_x=-1-speed_up+xiuzheng_x;
                    expectspeed_y=0+xiuzheng_y;
                }
                if(direction==1)
                {
                    expectspeed_x=0+xiuzheng_x;
                    expectspeed_y=-1-speed_up+xiuzheng_y;
                }
                if(direction==2)
                {
                    expectspeed_x=1+speed_up+xiuzheng_x;
                    expectspeed_y=0+xiuzheng_y;
                }
                if(direction==3)
                {
                    expectspeed_x=0+xiuzheng_x;
                    expectspeed_y=1+speed_up+xiuzheng_y;
                }
            }
            if(finding==1||finding==2)
            {
                expectspeed_x/=3.33;
                expectspeed_y/=3.33;
            }
        }

        /*******************发布期望速度***********************/




        auto target_speed_msg = std::make_shared<geometry_msgs::msg::TwistStamped>();
        target_speed_msg->twist.linear.x = expectspeed_x;
        target_speed_msg->twist.linear.y = expectspeed_y;
        target_speed_msg->header.stamp = this->get_clock()->now();
        speed_pub_->publish(*target_speed_msg);
    }

    // 云台角度回调
    void receiveCallback(const tdt_interface::msg::ReceiveData::SharedPtr msg)
    {
        pitch = msg->pitch;
        yaw = msg->yaw;
    }

    // 栅格地图回调
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        /*****************************保存或处理你的地图*****************************/
        int width = msg->info.width;
        int height = msg->info.height;
        cv::Mat map_image(height, width, CV_8UC1);
        map_height=height,map_width=width;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int index = x + y * width;
                int8_t occupancy_value = msg->data[index];
                uint8_t pixel_value = 0;

                if (occupancy_value == 0)
                    pixel_value = 255;
                else if (occupancy_value == 100)
                    pixel_value = 0;
                else
                    pixel_value = 128;

                map_image.at<uint8_t>(y, x) = pixel_value;
            }
        }
        usingmap=map_image;
        cv::imshow("Occupancy Grid Map", map_image);
        cv::waitKey(1);
    }


    void goalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        /***********************处理目标位置信息**************************/
        RCLCPP_INFO(this->get_logger(), "Goal position received: [x: %f, y: %f, z: %f]", msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }


    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr real_speed_sub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr speed_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr game_start_pub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr position_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_sub_;
    rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr send_data_pub_;
    rclcpp::Subscription<tdt_interface::msg::ReceiveData>::SharedPtr receive_data_sub_;
    float yaw = 0;
    float pitch = 0;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AutoShoot>());
    rclcpp::shutdown();
    return 0;
}