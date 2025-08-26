#include <thread>
#include <vector>

#include "bayesr.h"

//多线成执行函数
std::vector<BayesrResult> run_bayesr_chains(
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& y, 
    int num_chains = 10,
    int niter = 1000,
    int thin = 10,
    const Eigen::VectorXd& gamma = (Eigen::VectorXd(4) << 0, 0.01, 0.1, 1).finished(),
    const Eigen::VectorXd& startPi = (Eigen::VectorXd(4) << 0.5, 0.3, 0.15, 0.05).finished(),
    double startH2 = 0.5
) {
    //存储所有链的结果
    std::vector<BayesrResult> results(num_chains);
    //存储线程对象
    std::vector<std::thread> threads;

    //创建并启动所有线程
    for (int i = 0; i < num_chains; i++) {
        threads.emplace_back([&, i, thin]() {
            results[i] = bayesr(X, y, niter, 10, gamma, startPi, startH2); 
        });
    }
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }    
    return results;   
}

// 使用示例
int main() {
    // 假设以下是你的数据
    int n = 400;  // 样本量
    int p = 1000;  // 变量数
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, p);  // 随机生成X
    Eigen::VectorXd y = Eigen::VectorXd::Random(n);     // 随机生成y
    
    // 运行10条链
    int num_chains = 40;
    int niter = 20000;  // 迭代次数
    auto all_results = run_bayesr_chains(X, y, num_chains, niter);
    return 0;
}