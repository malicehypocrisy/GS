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

//自动识别有效应的SNP并估计其效应的大小
VectorXd rdirichlet(const VectorXd& alpha) {
  int k = alpha.size();
  VectorXd theta(k);
  double sum = 0.0;

  //使用Gammma分布抽取
  random_device rd;
  mt19937 gen(rd());

  for(int i = 0; i < k; i++) {
    gamma_distribution<double> gamma(alpha[i], 1.0);//狄利克雷分布可以通过伽马分布生成
    theta[i] = gamma(gen);
    sum += theta[i];
  }

  theta /=sum;
  return theta;
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

  // 存储MCMC结果
  MatrixXd keptIter(niter, ndist + 5);  // pi, nnz, sigmaSq, h2, vare, varg

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

      //计算概率
      double maxLog = logDelta.maxCoeff();//返回向量中最大的数
      VectorXd probDelta = (logDelta.array() - maxLog).exp();//计算e的幂
      probDelta /= probDelta.sum();//向量归一化

      //抽样选择分布
      double u = uniform_real_distribution<double>(0.0,1.0)(gen);
      int delta = 0;
      double sumProb = 0.0;
      for (; delta < ndist; delta++) {
        sumProb += probDelta[delta];
        if (sumProb >= u) break;//累计和sumProb大于等于随机数部分，此为选中的混合成分
      }

      nsnpDist[delta]++;

      if (delta > 0) {
        double newBeta = betaHat[delta] + sqrt(invLhs[delta])*normal(gen);
        ycorr += X.col(j) * (oldSample - newBeta);
        ghat += X.col(j) * newBeta;
        ssq += newBeta * newBeta / gamma[delta];
        beta[j] = newBeta;
        nnz++;
      } else {
        if (oldSample != 0) {
          ycorr += X.col(j) * oldSample;
        }
        beta[j] = 0;
      }
    }

    mcmc_beta.row(iter) = beta;

    //抽取pi
    pi = rdirichlet(nsnpDist.array() + 1.0);

    //抽样snp效应方差
    chisq_nnzub.param(chi_squared_distribution<double>::param_type(nnz + nub));//随机数复合自由度为nnz + nub的卡方分布 
    sigmaSq = (ssq + nub * scalee) / chisq_nnue(gen);

    //抽样残差方差
    double ycorr_sq = ycorr.squaredNorm();
    vare = (ycorr_sq + nue * scalee) / chisq_nnue(gen);

    //计算遗传力
    varg = var(ghat);
    h2 = varg / (varg + vare);

    //保存结果
    for (int k =0; k < ndist; k++) {
      keptIter(iter, k) = pi[k];
    }
    keptIter(iter, ndist) = nnz;
    keptIter(iter, ndist + 1) = sigmaSq;
    keptIter(iter, ndist + 2) = h2;
    keptIter(iter, ndist + 3) = vare;
    keptIter(iter, ndist + 4) = varg;

  }


}
 