#pragma once 
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <assert.h>
#include <chrono>
#include <limits>
#include <cmath>
#include <memory>
#include <functional>
#include <sys/timeb.h>
#include <ctime>
#include <fstream>

using namespace std;


using prec_t = double;
using numvec = vector<double>;
using indvec = vector<long>;
using sizvec = vector<size_t>;
constexpr prec_t THRESHOLD = 1e-6;
constexpr prec_t EPSILON = 1e-6;

//constexpr double EPSILON = 1e-5;


/**
 * Returns sorted indexes for the given array (in ascending order).
 */
template <typename T>
inline sizvec sort_indexes_ascending(vector<T> const& v) {
    // initialize original index locations
    vector<size_t> idx(v.size());
    // initialize to index positions
    iota(idx.begin(), idx.end(), 0);
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
    return idx;
}
template <typename T>
inline sizvec sort_indexes(std::vector<T> const& v) {
    // initialize original index locations
    sizvec idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i)
        idx[i] = i;
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
}

/**
 * Returns sorted indexes for the given array (in descending order).
 */
template <typename T>
inline sizvec sort_indexes_descending(vector<T> const& v) {
    // initialize original index locations
    vector<size_t> idx(v.size());
    // initialize to index positions
    iota(idx.begin(), idx.end(), 0);
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
    return idx;
}



/** This is useful functionality for debugging.  */
template<class T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    for (const auto& p : vec) {
        cout << p << " ";
    }
    return os;
}


/*
 This is a CLASS that contains all the information for problem instance
 */
class RMDPs_SA {
public:

    vector<vector<numvec>> P;       // transition kernel 
    vector<vector<numvec>> B;       // Branch location kernel
    vector<vector<numvec>> r;               // reward
    double gamma;                   // discount factor
    vector<double> rho;                       // initial distribution

    //double tau;                     // target
    //vector<double> w;                       // weights for smdps

    // auxiliary MDPs variables
    size_t nActions;      // get the number of actions
    size_t nStates;      // get the number of states

    // create auxiliary variables from input variables
    void createAuxiliaryVar() {
        // define nAction and nStates, and initialize pb_max and b_max_min
        nStates = P.size();
        nActions = P[0].size();
        //numvec w0(nStates, 1.0 / nStates);
        //w = w0;
    }
};


/*
 Belows are the function declaration.
 */
 //vector<int> P_branch(size_t nStates, size_t nBranchs);
RMDPs_SA Garnet_SARMDPs(size_t nStates, size_t nActions, size_t nBranchs);
vector<numvec> Rand_tolerance(const RMDPs_SA& prob);
vector<vector<prec_t>> Rand_Policy(const RMDPs_SA& prob);
pair<numvec, prec_t> worstcase_l1(numvec const& z, numvec const& pbar, prec_t const& xi);
//numvec RVI_SA(const RMDPs_SA& prob, const vector<numvec>& kappa);
tuple<vector<vector<prec_t>>, numvec, numvec> RVI_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const int& ite_time, const vector<prec_t>& v_ini);
pair<vector<vector<numvec>>, numvec> DLRPG_Innermax_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const vector<vector<prec_t>>& pi_t);
pair<vector<vector<numvec>>, numvec> DLRPG_RAAM_Innermax_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const vector<vector<prec_t>>& pi_t);
numvec MDP_Occu(const RMDPs_SA& prob, const vector<vector<numvec>>& P_now, const vector<vector<prec_t>>& pi_t_now);
vector<vector<prec_t>> MDP_grad_pi(const RMDPs_SA& prob, const vector<vector<numvec>>& P_now, const numvec& eta, const numvec& v_now);
vector<vector<prec_t>> DLRPG_outermin_SA(const RMDPs_SA& prob, const vector<vector<prec_t>>& pi_old, const double& step, const vector<vector<prec_t>>& G_now);
//pair<vector<vector<prec_t>>, numvec> DLRPG_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const double& step, const int& Ite_time);
double L2_Gap(const numvec& v1, const numvec& v2);
double inner_product(const numvec& v1, const numvec& v2);
pair<vector<vector<prec_t>>, numvec> DLRPG_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const double& step, const numvec& v_opt);
pair<vector<vector<prec_t>>, numvec> DLRPG_RAAM_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const double& step, const numvec& v_opt);
//tuple<prec_t, prec_t, prec_t, prec_t, prec_t, prec_t, prec_t, prec_t, prec_t, prec_t> get_speed_gurobi(RSMDPs& prob, const size_t nStates, const size_t instance_num);
//void runsave_rsmdps_speed(const function<RSMDPs(size_t nStates, size_t nActions)>& prob_gen, const sizvec nStates_ls, const size_t repetitions);
 /* experiment */
pair<vector<vector<numvec>>, numvec> DLRPG_PGD_Innermax_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const vector<vector<prec_t>>& pi_t, const vector<vector<numvec>>& P_ini, const double& step);
vector<vector<numvec>> MDP_grad_P(const RMDPs_SA& prob, const vector<vector<prec_t>>& pi_now, const vector<vector<numvec>>& P_now, const numvec& eta, const numvec& v_now);
numvec MDP_value(const RMDPs_SA& prob, const vector<vector<prec_t>>& pi_now, const vector<vector<numvec>>& P_now);
tuple<vector<vector<prec_t>>, numvec, numvec> DLRPG_PGD_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const double& out_step, const double& in_step, const int& ite_time);
void Sample_sa(const int& Garnet_Snum, const int& Garnet_Anum, const int& Garnet_Bnum, const int& itenum, const double& out_step, const double& in_step, const int& sample_num);