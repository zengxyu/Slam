//
// Created by zeng on 31.03.20.
// Use ceres library to solve non-linear optimization
//

#include <iostream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace ceres;

// 第一部分：构建代价函数
struct CURVE_FITTING_COST_FUNCTION{
    CURVE_FITTING_COST_FUNCTION(double x,double y):_x(x),_y(y){}
    template <typename T>
    bool operator()(const T* const abc, T* residual)const {
        residual[0] = T(_y) - exp(abc[0] * pow(T(_x),2) + abc[1]*_x+abc[2]);
        return true;
    }
    const double _x;
    const double _y;
};

int main(int argc, char **argv){
    //    生成數據
    double a = 3, b = 2,c = 1;
    double w = 1;
    RNG rng;
    double abc[3] = {0,0,0};
    vector<double> x_data,y_data;
    for (int i =0;i<1000;i++){
        double x = i/1000.0;
        x_data.push_back(x);
        y_data.push_back(exp(a*pow(x,2) + b*x + c) + rng.gaussian(w));
    }
    //第二部分, 构建寻优问题
    Problem problem;
    for (int i =0;i<1000;i++){
        CostFunction* costFunction = new ceres::AutoDiffCostFunction<CURVE_FITTING_COST_FUNCTION,1,3>(
                new CURVE_FITTING_COST_FUNCTION(x_data[i],y_data[i]));//使用自動求導
        problem.AddResidualBlock(costFunction,nullptr,abc);//向問題中添加誤差項
    };
    //第三部分, 配置并运行求解器
    Solver::Options options;
    options.linear_solver_type = DENSE_QR;//配置增量方程的解法
    options.minimizer_progress_to_stdout = true;//输出到cout
    Solver::Summary summary;//优化信息
    ceres::Solve(options,&problem,&summary);//求解
    cout<<"a:" <<abc[0] << endl;
    cout<<"b:" <<abc[1] << endl;
    cout<<"c:" <<abc[2] << endl;
    return 0;
}
