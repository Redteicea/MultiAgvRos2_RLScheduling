#include "fleet_manager/rl_policy_copy.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace fleet_manager
{
    static constexpr int N_SITES_C = 24;                    // 站点数 = len(SITE_COORD)
    static constexpr int SLOT_C = 3 + 2 * N_SITES_C + 1;    // 每任务槽 = 52

    RLPolicy2::RLPolicy2(const std::string& model_path, int n_agents): n_agents_(n_agents)
    {
        if (n_agents_ != NUM_AGENTS)
        {
            throw std::runtime_error("RLPolicy: 车队规模须为 " + std::to_string(NUM_AGENTS) +
            "（策略按 3 车训练），实际 " + std::to_string(n_agents_));
        }
        try {
            model_ = torch::jit::load(model_path);
        } 
        catch (const exception& e){
            throw std::runtime_error("RLPolicy: 加载模型失败 " + std::string(e.what()));
        }
        model_.eval();
    }

    
}