#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iomanip>
#include <cassert>

using namespace Eigen;
using namespace std;

//存储BayesR后验结果的结构体
struct BayesrResult {
    VectorXd pi_mean; //pi的后验均值
    double nnz_mean; //非零效应数的后验均值
    double sigmaSq_mean; //sigmaSq的后验均值
    double h2_mean;         // 遗传力的后验均值
    double vare_mean;       // 残差方差的后验均值
    double varg_mean;       // 遗传方差的后验均值
    VectorXd beta_mean;     // SNP效应的后验均值
};
 
//计算矩阵方差的函数
double var(const VectorXd& mat, int df = 1) {
    if(mat.size() == 0) return 0.0;

    double mean = mat.mean();
    double sumSquared = (mat.array() - mean).square().sum();
    return sumSquared / (mat.size() - df); 
} 

// 避免数据溢出
double log_sum_exp(const VectorXd& log_values) {
    double max_val = log_values.maxCoeff();
    return max_val +log((log_values.array() - max_val).exp().sum());
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
    if (sum < 1e-10) {
        theta.fill(1.0 / k);
    } else {
        theta /= sum;
    }   
    return theta;
}

 //贝叶斯回归主函数
BayesrResult bayesr(
    const MatrixXd& X, 
    const VectorXd& y, 
    int niter = 1000,
    int thin = 10, //每隔10保存1次
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

    //方差相关参数
    double vary = var(y);
    double varg = vary * h2;
    double vare = vary * (1 - h2);

    double gamma_pi_sum = (gamma.array() * pi.array()).sum();
    double sigmaSq = varg / (m * gamma_pi_sum);

    const int nub = 4; //SNP效应方差的先验的自由度
    const int nue = 4; //残差方差的先验自由度
    double scaleb = (nub - 2) / nub * sigmaSq; //SNP效应方差先验尺度
    double scalee = (nue - 2) / nue * vare; //残差方差的先验尺度

    //初始化存储变量
    VectorXd beta = VectorXd::Zero(m);  // 当前迭代的SNP效应
    VectorXd mu = VectorXd::Constant(n, y.mean());  // 截距向量
    VectorXd ycorr = y - mu;  // 表型残差（y - 截距 - 遗传效应）
    VectorXd xpx(m);          // X'X的对角元素

    for (int j = 0; j < m; j++) {
        xpx[j] = X.col(j).squaredNorm();
    }

    VectorXd pi_sum = VectorXd::Zero(ndist);          // pi的累加和
    double nnz_sum = 0.0;                             // 非零效应数的累加和
    double sigmaSq_sum = 0.0;                         // sigmaSq的累加和
    double h2_sum = 0.0;                              // h2的累加和
    double vare_sum = 0.0;                            // vare的累加和
    double varg_sum = 0.0;                            // varg的累加和
    VectorXd beta_sum = VectorXd::Zero(m);

    //随机数生成器
    mt19937 gen(1234);
    normal_distribution<double> normal(0.0, 1.0);
    uniform_real_distribution<double> uniform(0.0, 1.0);//[0,1]均匀分布

    //MCMC主循环
    for (int iter = 0; iter < niter; iter++) {
        //抽样截距
        ycorr += mu;  // 恢复表型：ycorr = y - 遗传效应 → y - 遗传效应 + mu = y - (遗传效应 - mu)
        double mu_mean = ycorr.mean();  // 截距的后验均值
        double mu_var = vare / n;       // 截距的后验方差
        double mu_scalar = mu_mean + sqrt(mu_var) * normal(gen);  // 从正态分布抽样截距
        mu.fill(mu_scalar);  // 更新截距向量
        ycorr -= mu;         // 重新计算残差：ycorr = y - mu - 遗传效应

        VectorXd invSigmaSq = 1.0 / (gamma.array() * sigmaSq);
        VectorXd logSigmaSq = (gamma.array() * sigmaSq ).log(); 
        VectorXd logPi = pi.array().log();

        VectorXd nsnpDist = VectorXd::Zero(ndist);//统计每个混合分布被选中SNP的数量 
        double ssq = 0.0;//累计SNP效应的加权平方和
        int nnz = 0;//统计非零效应的SNP
        VectorXd ghat = VectorXd::Zero(n);//存储每个样本的遗传效应的预测值

        // 抽样SNP效应
        for (int j = 0; j < m; j++) {
            double old_beta = beta[j];
            double rhs_beta = X.col(j).dot(ycorr) + xpx[j] * old_beta;//更新残差
            rhs_beta /= vare;
              
            VectorXd beta_hat(ndist);    // 各成分的后验均值
            VectorXd beta_var_inv(ndist); // 各成分的后验方差倒数
              
            for (int k = 0; k < ndist; k++) {
                beta_var_inv[k] = xpx[j] / vare + invSigmaSq[k];
                beta_hat[k] = rhs_beta / beta_var_inv[k]; 
              } 

            // 混合分布抽样
            VectorXd log_delta(ndist);
            for (int k = 0; k < ndist; k++) {
                if (k == 0) {
                  // k=0时效应为0，只需先验概率
                  log_delta[k] = logPi[k];
                } else {
                    // 完整的对数后验概率计算
                    log_delta[k] = 0.5 * (log(1 / beta_var_inv[k]) - logSigmaSq[k] + beta_hat[k] *  beta_hat[k] * beta_var_inv[k]) + logPi[k];                             
                  }
              }
            //使用log-sum-exp技巧进行归一化
            double log_sum = log_sum_exp(log_delta);
            VectorXd prob_delta = (log_delta.array() - log_sum).exp();

            //抽样选择分布
            double u = uniform(gen);
            int delta = 0;
            double cum_prob = 0.0;
            for (; delta < ndist - 1; delta++) {
                cum_prob += prob_delta[delta];
                  if (u <= cum_prob) break;
          }
          nsnpDist[delta]++;

          if (delta > 0) {
            double new_beta = beta_hat[delta] + sqrt(1.0 / beta_var_inv[delta]) * normal(gen);
            ycorr += X.col(j) * (old_beta - new_beta); 
            ghat += X.col(j) * new_beta;
            ssq += new_beta * new_beta / gamma[delta];
            beta[j] = new_beta;
            nnz++;
          } else {
            if (old_beta != 0) {
              ycorr += X.col(j) * old_beta;
            }
            beta[j] = 0;
          }
      }

      //抽取pi
      pi = rdirichlet(nsnpDist.array()+ 1e-10, gen);

      //抽样snp效应方差
      int df_sigmaSq = nnz + nub;  // 卡方分布自由度
      chi_squared_distribution<double> chisq_sigmaSq(df_sigmaSq);
      sigmaSq = (ssq + nub * scaleb) / chisq_sigmaSq(gen);  // 逆卡方抽样公式
      sigmaSq = max(sigmaSq, 1e-10);  // 避免方差过小导致数值不稳定

      //抽样残差方差
      int df_vare = n + nue;  // 卡方分布自由度
      double ycorr_sq = ycorr.squaredNorm();  // 残差平方和
      chi_squared_distribution<double> chisq_vare(df_vare);
      vare = (ycorr_sq + nue * scalee) / chisq_vare(gen);
      vare = max(vare, 1e-10);  // 避免方差过小

      //计算遗传力
      varg = var(ghat);
      h2 = varg / (varg + vare);
      h2 = clamp(h2, 0.0, 1.0); 

      //保存结果
      if ((iter + 1) % thin == 0) {  // 仅在thin的倍数迭代时累加
          pi_sum += pi;
          nnz_sum += nnz;
          sigmaSq_sum += sigmaSq;
          h2_sum += h2;
          vare_sum += vare;
          varg_sum += varg;
          beta_sum += beta;

        // 打印迭代过程
        if (iter % 100 == 0) {  // 每100次有效迭代打印一次
            cout << "Iter: " << setw(5) << (iter + 1)
                << " | NNZ: " << setw(4) << nnz
                << " | sigmaSq: " << setw(6) << fixed << setprecision(4) << sigmaSq
                << " | h2: " << setw(6) << fixed << setprecision(4) << h2
                << " | vare: " << setw(6) << fixed << setprecision(4) << vare
                << " | varg: " << setw(6) << fixed << setprecision(4) << varg << endl;
        }
      }
    } 

    // 计算后验均值
    int n_saved = niter / thin;
    BayesrResult result;
    result.pi_mean = pi_sum / n_saved;
    result.nnz_mean = nnz_sum / n_saved;
    result.sigmaSq_mean = sigmaSq_sum / n_saved;
    result.h2_mean = h2_sum / n_saved;
    result.vare_mean = vare_sum / n_saved;
    result.varg_mean = varg_sum / n_saved;
    result.beta_mean = beta_sum / n_saved;  
 
    // 打印后验均值
    std::cout << "\nPosterior mean:" << endl;
    std::cout << "Pi: ";
    for (int k = 0; k < ndist; k++) {
        std::cout << fixed << setprecision(4) << result.pi_mean[k] << " ";
    }
    std::cout << "\nnnz: " << fixed << setprecision(1) << result.nnz_mean
              << ", sigmaSq: " << fixed << setprecision(6) << result.sigmaSq_mean
              << ", h2: " << fixed << setprecision(4) << result.h2_mean
              << ", vare: " << fixed << setprecision(6) << result.vare_mean
              << ", varg: " << fixed << setprecision(6) << result.varg_mean << endl;

    return result;
}


 