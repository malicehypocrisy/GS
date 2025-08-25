#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iomanip>
#include <numeric>

using namespace Eigen;
using namespace std;

 
//计算矩阵方差的函数
double var(const VectorXd& mat, int df = 0)
{
  
  if(mat.size() == 0) return 0.0;

  double mean = mat.mean();
  double sumSquared = (mat.array() - mean).square().sum();
  return sumSquared / (mat.size() - df); 
}

 //贝叶斯回归主函数
pair <MatrixXd, MatrixXd> bayesr(const MatrixXd& X, const VectorXd& y, int niter = 1000,
                                const VectorXd& gamma= VectorXd::Zero(4),
                                const VectorXd& startPi = VectorXd::Zero(4),
                                double startH2 = 0.5) 
{
  int n = X.rows(); //观测数
  int m  = X.cols(); //snp数量
  int ndist = startPi.size(); //混合分布的数量

  //初始化参数
  VectorXd pi = startPi;
  double h2 = startH2;
  double vary = var(y);
  double varg = vary * h2;
  double vare = vary * (1 - h2);
  double sigmaSq = varg / (m * (gamma.array() * pi.array()).sum());//SNP的方差

  const int nub = 4; //SNP效应方差的先验的自由度
  const int nue = 4; //残差方差的先验自由度
  double scaleb = (nub - 2) / nub * sigmaSq; //SNP效应方差先验尺度
  double scalee = (nue - 2) / nue * vare; //残差方差的先验尺度

  VectorXd beta = VectorXd::Zero(m);//存储SNP效应
  MatrixXd mcmc_beta(niter, m);

  double mu_scalar = y.mean(); 
  VectorXd mu = VectorXd::Constant(n, mu_scalar);//c++不允许，在同一个作用域中有两个同名变量
  VectorXd ycorr = y - mu;

  //计算X`X的对角元素
  VectorXd xpx(m);
  for (int j = 0; j < m; j++) {
    xpx[j] = X.col(j).squaredNorm();
  }

  //随机数生成器
  random_device rd;//真随机数生成器（依赖硬件与操作系统提供的随机源）
  mt19937 gen(rd());//初始化随机数引擎
  normal_distribution<double> normal(0.0, 1.0);
  chi_squared_distribution<double> chisq_nnzub(0);//snp的初始自由度
  chi_squared_distribution<double> chisq_nnue(n + nue);//残差的自由度

  //MCMC主循环
  for (int iter = 0; iter < niter; niter++) {
    //抽样截距
    ycorr += mu;//恢复表型
    double rhs = ycorr.sum();
    double invLhs = 1.0 / n;
    double muhat = invLhs * rhs;
    mu_scalar = muhat + sqrt(invLhs * vare) * normal(gen);//从正态分布中抽样一个均值
    ycorr -= mu;

    //抽样SNP效应
    VectorXd logPi = pi.array().log();
    VectorXd logPiComp = (1 - pi.array()).log();
    VectorXd invSigmaSq = 1.0 / (gamma.array() * sigmaSq).array();
    VectorXd logSigmaSq = (gamma.array() *sigmaSq).log().array();

    VectorXd nsnpDist = VectorXd::Zero(ndist);//统计每个混合分布被选中SNP的数量 
    double ssq = 0.0;//累计SNP效应的加权平方和
    int nnz = 0;//统计非零效应的SNP
    VectorXd ghat = VectorXd::Zero(n);//存储每个样本的遗传效应的预测值

    for (int j =0; j < m; j++) {
      double oldSample = beta[j];
      double rhs = X.col(j).dot(ycorr) + xpx[j] * oldSample;
      rhs /= vare;

      VectorXd invLhs = 1.0 / (xpx[j] / vare + invSigmaSq.array()).array();
      VectorXd betaHat = invLhs * rhs;

      //抽样混合分布成员
      VectorXd logDelta(ndist);
      for (int k = 0; k < ndist; k++) {
        logDelta[k] = 0.5 * (log(invLhs[k]) - logSigmaSq[k] + betaHat[k] * rhs) + logPi[k];
      }


    }
  }


}
 