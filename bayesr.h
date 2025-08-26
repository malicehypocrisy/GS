#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iomanip>
#include <numeric>

using namespace Eigen;
using namespace std;

 
//计算矩阵方差的函数
double var(const VectorXd& mat, int df = 1)
{ 
  if(mat.size() == 0) return 0.0;

  double mean = mat.mean();
  double sumSquared = (mat.array() - mean).square().sum();
  return sumSquared / (mat.size() - df); 
}

//生成狄利克雷分布随机向量
VectorXd rdirichlet(const VectorXd& alpha, mt19937& gen) {
  int k = alpha.size();
  VectorXd theta(k);
  double sum = 0.0;

  for(int i = 0; i < k; i++) {
    gamma_distribution<double> gamma(alpha[i], 1.0);//狄利克雷分布可以通过伽马分布生成
    theta[i] = gamma(gen);
    sum += theta[i];
  }
  theta /=sum;//归一化得到狄利克雷分布
  return theta;
}

 //贝叶斯回归主函数
pair <MatrixXd, MatrixXd> bayesr(
    const MatrixXd& X, 
    const VectorXd& y, 
    int niter = 1000,
    const VectorXd& gamma= (VectorXd(4) <<0, 0.01, 0.1, 1).finished(),
    const VectorXd& startPi = (VectorXd(4) <<0.5, 0.3, 0.15, 0.05).finished(),
    double startH2 = 0.5
) {

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
  //初始化存储变量
  VectorXd beta = VectorXd::Zero(m);//存储SNP效应
  MatrixXd mcmc_beta(niter, m);//存储所有迭代的beta值
  double mu_scalar = y.mean(); 
  VectorXd mu = VectorXd::Constant(n, mu_scalar);
  VectorXd ycorr = y - mu;

  //计算X`X的对角元素
  VectorXd xpx(m);
  for (int j = 0; j < m; j++) {
    xpx[j] = X.col(j).squaredNorm();
  }

  // 存储MCMC结果
  MatrixXd keptIter(niter, ndist + 5);  // pi, nnz, sigmaSq, h2, vare, varg

  //随机数生成器
  mt19937 gen;
  gen.seed(1234);

  normal_distribution<double> normal(0.0, 1.0);
  uniform_real_distribution<double> uniform(0.0, 1.0);//[0,1]均匀分布

  //MCMC主循环
  for (int iter = 0; iter < niter; iter++) {
    //抽样截距
    ycorr += mu;//恢复表型
    double rhs = ycorr.sum();
    double invLhs = 1.0 / n;
    double muhat = invLhs * rhs;
    mu_scalar = muhat + sqrt(invLhs * vare) * normal(gen);//从正态分布中抽样一个均值
    mu = VectorXd::Constant(n, mu_scalar);
    ycorr = y - mu;//重新调整表型（去除新截距）

    //抽样SNP效应
    VectorXd logPi = pi.array().log();
    VectorXd invSigmaSq = 1.0 / (gamma.array() * sigmaSq);
    VectorXd logSigmaSq = (gamma.array() *sigmaSq).log();

    VectorXd nsnpDist = VectorXd::Zero(ndist);//统计每个混合分布被选中SNP的数量 
    double ssq = 0.0;//累计SNP效应的加权平方和
    int nnz = 0;//统计非零效应的SNP
    VectorXd ghat = VectorXd::Zero(n);//存储每个样本的遗传效应的预测值

    for (int j =0; j < m; j++) {
      double oldSample = beta[j];
      double rhs_beta = X.col(j).dot(ycorr) + xpx[j] * oldSample;
      rhs_beta /= vare;

      VectorXd invLhs_beta = 1.0 / (xpx[j] / vare + invSigmaSq.array()).array();
      VectorXd betaHat = invLhs_beta.array() * rhs_beta;

      //抽样混合分布成员
      VectorXd logDelta(ndist);
      for (int k = 0; k < ndist; k++) {
        if (k == 0) {
          logDelta[k] = logPi[k];
        } else {
          logDelta[k] = 0.5 * (log(invLhs_beta[k]) - logSigmaSq[k] + betaHat[k] * rhs_beta) + logPi[k];
        }
      }

      //概率归一化
      double maxLog = logDelta.maxCoeff();//返回向量中最大的数
      VectorXd probDelta = (logDelta.array() - maxLog).exp();//计算e的幂
      probDelta /= probDelta.sum();//向量归一化

      //抽样选择分布
      double u = uniform(gen);
      int delta = 0;
      double sumProb = 0.0;
      for (; delta < ndist; delta++) {
        sumProb += probDelta[delta];
        if (sumProb >= u) break;//累计和sumProb大于等于随机数部分，此为选中的混合成分
      }

      nsnpDist[delta]++;

      if (delta > 0) {
        double newBeta = betaHat[delta] + sqrt(invLhs_beta[delta]) * normal(gen);
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
    pi = rdirichlet(nsnpDist.array() + 1.0, gen);

    //抽样snp效应方差
    chi_squared_distribution<double> chisq_nnzub(nnz + nub);//随机数复合自由度为nnz + nub的卡方分布 
    sigmaSq = (ssq + nub * scaleb) / chisq_nnzub(gen);

    //抽样残差方差
    double ycorr_sq = ycorr.squaredNorm();
    chi_squared_distribution<double> chisq_nnue(n + nue);
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

    //每100次迭代输出一次信息
    if((iter + 1) % 100 == 0) {
      std::cout << "\n iter" << setw(4) << (iter + 1)
           << ", nnz =" <<setw(4) <<nnz
           << ", sigmasq =" << setw(6) << fixed << setprecision(3) << sigmaSq
           << ", h2 =" << setw(6) << fixed << setprecision(3) << h2
           << ", vare =" << setw(6) << fixed << setprecision(3) << vare
           << ", varg =" << setw(6) << fixed << setprecision(3) << varg << endl;
    }
  }
  //计算后验均值
  VectorXd postMean(ndist + 5);
  for (int k = 0; k < ndist + 5; k++) {
    postMean[k] = keptIter.col(k).mean();
  }

  std::cout << "\nPosterior mean:" << endl;
  std::cout << "Pi: ";
  for (int k = 0; k < ndist; k++) {
    std::cout << fixed << setprecision(4) << postMean[k] << " ";
    }
  std::cout << "\nnnz: " << fixed << setprecision(1) << postMean[ndist]
            << ", sigmaSq: " << fixed << setprecision(4) << postMean[ndist+1]
            << ", h2: " << fixed << setprecision(4) << postMean[ndist+2]
            << ", vare: " << fixed << setprecision(4) << postMean[ndist+3]
            << ", varg: " << fixed << setprecision(4) << postMean[ndist+4] << endl;

  return {keptIter, mcmc_beta};
}


 