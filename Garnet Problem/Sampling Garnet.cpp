#include "Sampling Garnet.h"
#include <random>
#include <fstream>
#include <functional>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <assert.h>
#include <chrono>
#include <limits>
#include <cmath>
#include <memory>
#include <stdio.h>
#include "gurobi_c++.h"
#include <sstream>
#include "Eigen\Dense"
using namespace Eigen;


/*********************************************
Generate a random Garnet for Robust_MDPs_SA/ --- for S-rectangular, the nominal transition will be modified
**************************************************/

// This is a function for randomly selecting n number from 0-100 and ordering it
vector<int> P_branch(size_t nStates, size_t nBranchs) {
    srand(unsigned(time(NULL)));
    vector<int> S_temp; S_temp.reserve(nStates);
    vector<int> b_temp; b_temp.reserve(nBranchs);
    for (size_t s = 0; s < nStates; s++) {
        S_temp.push_back(s);
    }
    //std::shuffle(S_temp.begin(), S_temp.end());
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(S_temp.begin(), S_temp.end(), default_random_engine(seed));
    for (size_t b = 0; b < nBranchs; b++) {
        b_temp.push_back(S_temp[b]);
    }
    //sort(b_temp.begin(), b_temp.end());
    return b_temp;
}

// Set Garnet problem
RMDPs_SA Garnet_SARMDPs(size_t nStates, size_t nActions, size_t nBranchs) {

    // Create random number generator with default seed
    default_random_engine generator;

    generator.seed(chrono::system_clock::now().time_since_epoch().count());
    // comment this if you need the same instance all the time.
    uniform_real_distribution<double> distribution(0.0, 5.0);

    double gamma = 0.95;


    //randomized initial distribution
    numvec rho; rho.reserve(nStates);
    double rho_sum = 0.0;
    for (size_t s = 0; s < nStates; s++) {
        prec_t rho_temp = distribution(generator);
        rho.push_back(rho_temp);
        rho_sum += rho_temp;
    }
    for (size_t s = 0; s < nStates; s++) {
        rho[s] = rho[s] / rho_sum;
    }

    //reward r(s,a,s')
    vector<vector<numvec>> r; r.reserve(nStates);
    for (size_t s = 0; s < nStates; s++) {
        vector<numvec> r_s; r_s.reserve(nActions);
        for (size_t a = 0; a < nActions; a++) {
            numvec r_sa; r_sa.reserve(nStates);
            for (size_t s2 = 0; s2 < nStates; s2++) {
                r_sa.push_back(distribution(generator));
            }
            r_s.push_back(r_sa);
        }
        r.push_back(r_s);
    }

    // nominal transition and branch location kernel
    vector<vector<numvec>> P;
    P.reserve(nStates);
    vector<vector<numvec>> B;
    B.reserve(nStates);
    for (size_t s = 0; s < nStates; s++) {
        vector<numvec> P_s;
        P_s.reserve(nActions);
        vector<numvec> B_s;
        B_s.reserve(nActions);
        for (size_t a = 0; a < nActions; a++) {
            double P_sa_sum = 0.0;
            numvec P_sa;
            numvec B_sa;
            vector<int> index = P_branch(nStates, nBranchs);
            P_sa.reserve(nStates);
            P_sa.assign(nStates, 0.0);
            B_sa.reserve(nStates);
            B_sa.assign(nStates, 1.0);
            for (size_t b = 0; b < nBranchs; b++) {
                prec_t p_temp = distribution(generator);
                P_sa[index[b]] = p_temp;
                B_sa[index[b]] = 0.0;
                P_sa_sum += p_temp;
            }
            for (size_t s2 = 0; s2 < nStates; s2++) {
                P_sa[s2] = P_sa[s2] / P_sa_sum;
            }
            P_s.push_back(P_sa);
            B_s.push_back(B_sa);
        }
        P.push_back(P_s);
        B.push_back(B_s);
    }
    RMDPs_SA instance;
    instance.P = P;
    instance.B = B;
    instance.r = r;
    instance.rho = rho;
    instance.gamma = gamma;
    instance.createAuxiliaryVar();

    return instance;
}

// Determine a random tolerance kappa
vector<numvec> Rand_tolerance(const RMDPs_SA& prob) {
    vector<numvec> kappa; kappa.reserve(prob.nStates);

    default_random_engine rand;

    rand.seed(chrono::system_clock::now().time_since_epoch().count());
    // comment this if you need the same instance all the time.
    uniform_real_distribution<double> distribution(0.1, 0.5);

    for (size_t s = 0; s < prob.nStates; s++) {
        numvec kappa_s; kappa_s.reserve(prob.nActions);
        for (size_t a = 0; a < prob.nActions; a++) {
            kappa_s.push_back(distribution(rand));
        }
        kappa.push_back(kappa_s);
    }
    return kappa;
}

/***************************************************************
Robust Value Iteration for SA-rectangularity RMDP (pi, v_opt)
**************************************************************/


// One contraction mapping
pair<vector<vector<prec_t>>, numvec> gurobi_L1RVI_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const numvec v_start)
{
    // general constants values
    constexpr double inf = numeric_limits<double>::infinity();
    const vector<vector<numvec>>& P = prob.P;
    const vector<vector<numvec>>& B = prob.B;
    const vector<vector<numvec>>& r = prob.r;
    const numvec& rho = prob.rho;
    const prec_t& gamma = prob.gamma;
    const size_t nStates = P.size();
    const size_t nActions = P[0].size();

    // set updated value function
    numvec v_finish(nStates, 0.0);
    // Initialize policy
    vector<vector<prec_t>> pi; pi.reserve(nStates);
    for (int s_0 = 0; s_0 < nStates; s_0++)
    {
        numvec pi_s(nActions, 0.0);
        pi.push_back(pi_s);
    }

    for (int s = 0; s < nStates; s++)
    {
        int temp_Amin = 0;
        double temp_min = 0.0;
        for (int a = 0; a < nActions; a++)
        {
            numvec temp_c; temp_c.reserve(nStates);
            for (int s_1 = 0; s_1 < nStates; s_1++)
            {
                //temp_c[s_1] = r[s][a][s_1] + gamma * v_0[s_1];
                temp_c.push_back(r[s][a][s_1] + gamma * v_start[s_1]);
            }

            GRBEnv env = GRBEnv(true);
            env.set(GRB_IntParam_OutputFlag, 0);
            //env.set(GRB_IntParam_Threads, 1);
            env.start();
            GRBModel model = GRBModel(env);

            // Define decision variables
            GRBVar* p_sa;
            p_sa = model.addVars(nStates, GRB_CONTINUOUS);
            for (int i = 0; i < nStates; i++)
            {
                p_sa[i].set(GRB_DoubleAttr_LB, 0.0);
                p_sa[i].set(GRB_DoubleAttr_UB, 1.0);
            }
            GRBVar* temp_y;
            temp_y = model.addVars(nStates, GRB_CONTINUOUS);
            for (int i = 0; i < nStates; i++)
            {
                temp_y[i].set(GRB_DoubleAttr_LB, -inf);
                temp_y[i].set(GRB_DoubleAttr_UB, inf);
            }

            // objective
            GRBLinExpr objective;
            // constraints
            GRBLinExpr sum_p;
            GRBLinExpr sum_y;
            GRBLinExpr sum_bp;
            // p_sa^T 1 = 1
            for (int s_3 = 0; s_3 < nStates; s_3++)
            {
                sum_p += p_sa[s_3];
            }
            model.addConstr(sum_p == 1.0);
            // y^T 1 = 1
            for (int s_4 = 0; s_4 < nStates; s_4++)
            {
                sum_y += temp_y[s_4];
            }
            model.addConstr(sum_y <= kappa[s][a]);

            // y = |p - p_c|
            for (int s_5 = 0; s_5 < nStates; s_5++)
            {
                model.addConstr(p_sa[s_5] <= temp_y[s_5] + P[s][a][s_5]);
                model.addConstr(p_sa[s_5] >= P[s][a][s_5] - temp_y[s_5]);
            }

            // branch location constraint

            for (int s_6 = 0; s_6 < nStates; s_6++)
            {
                sum_bp += p_sa[s_6] * B[s][a][s_6];
            }
            model.addConstr(sum_bp == 0.0);
            // set objective
            for (int s_7 = 0; s_7 < nStates; s_7++) {
                objective += p_sa[s_7] * temp_c[s_7];
            }
            model.setObjective(objective, GRB_MAXIMIZE);

            // run optimization
            model.optimize();
            double Inner_P = model.get(GRB_DoubleAttr_ObjVal);
            if (a == 0)
            {
                temp_min = Inner_P;
            }
            else
            {
                if (temp_min >= Inner_P)
                {
                    temp_min = Inner_P;
                    temp_Amin = a;
                }
            }
        }
        pi[s][temp_Amin] = 1.0;
        v_finish[s] = temp_min;
    }
    return { pi, v_finish };
}


// Apply Contraction to solve robust optimal value funtion
tuple<vector<vector<prec_t>>, numvec, numvec> RVI_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const int& ite_time, const vector<prec_t>& v_ini)
{
    const size_t nStates = prob.P.size();
    const size_t nActions = prob.P[0].size();


    // set zero vector as the initial v_0
    numvec v_0 = v_ini;
    // set temp vector
    numvec v_1(nStates, 0.0);
    int n = 0;
    numvec J; J.reserve(ite_time);
    while (true)
    {
        n += 1;
        auto [pi_now, v_1] = gurobi_L1RVI_SA(prob, kappa, v_0);
        double J_ite = inner_product(v_1, prob.rho);
        J.push_back(J_ite);
        if (n == ite_time)
        {
            return { pi_now, v_1, J };
            break;
        }
        v_0 = v_1;
    }
}

/***************************************************************
DL-RPG for SA-rectangularity RMDP (pi, v_opt)
**************************************************************/

// Initial a random policy 
vector<vector<prec_t>> Rand_Policy(const RMDPs_SA& prob)
{
    default_random_engine rand;
    rand.seed(chrono::system_clock::now().time_since_epoch().count());
    uniform_real_distribution<double> distribution(0.1, 0.5);

    vector<vector<prec_t>> pi; pi.reserve(prob.nStates);
    for (int s_0 = 0; s_0 < prob.nStates; s_0++) {
        numvec pi_s; pi_s.reserve(prob.nActions);
        double temp_sum = 0.0;
        for (int a_0 = 0; a_0 < prob.nActions; a_0++)
        {
            double temp = distribution(rand);
            pi_s.push_back(temp);
            temp_sum += temp;
        }
        for (int a_1 = 0; a_1 < prob.nActions; a_1++)
        {
            pi_s[a_1] = pi_s[a_1] / temp_sum;
        }
        pi.push_back(pi_s);
    }
    return pi;
}


// Compute the occupancy measure
numvec MDP_Occu(const RMDPs_SA& prob, const vector<vector<numvec>>& P_now, const vector<vector<prec_t>>& pi_t_now)
{

    vector<vector<prec_t>> P_pi; P_pi.reserve(prob.nStates);
    for (int s = 0; s < prob.nStates; s++) {
        numvec P_pi_s; P_pi_s.reserve(prob.nStates);
        for (int s_prime = 0; s_prime < prob.nStates; s_prime++) {
            double temp = 0;
            for (int a = 0; a < prob.nActions; a++)
            {
                temp += P_now[s][a][s_prime] * pi_t_now[s][a];
            }
            P_pi_s.push_back(temp);
        }
        P_pi.push_back(P_pi_s);
    }

    MatrixXd mat_P_pi(prob.nStates, prob.nStates);
    for (int s = 0; s < prob.nStates; s++)
    {
        for (int s_prime = 0; s_prime < prob.nStates; s_prime++)
        {
            mat_P_pi(s, s_prime) = P_pi[s][s_prime];
        }
    }
    VectorXd b_right(prob.nStates);
    for (int s = 0; s < prob.nStates; s++)
    {
        b_right(s) = (1 - prob.gamma) * prob.rho[s];
    }

    MatrixXd I = MatrixXd::Identity(prob.nStates, prob.nStates);

    MatrixXd mat_temp1(prob.nStates, prob.nStates);
    mat_temp1 = I - prob.gamma * mat_P_pi;

    VectorXd x = mat_temp1.colPivHouseholderQr().solve(b_right);
    numvec Eta(prob.nStates, 0.0);
    for (int s = 0; s < prob.nStates; s++)
    {
        Eta[s] = x(s);
    }
    return Eta;

}

//RAAM inner problem
pair<numvec, prec_t> worstcase_l1(numvec const& z, numvec const& pbar, prec_t const& xi) {
    assert(*min_element(pbar.cbegin(), pbar.cend()) >= -THRESHOLD);
    assert(*max_element(pbar.cbegin(), pbar.cend()) <= 1 + THRESHOLD);
    assert(xi >= -EPSILON);
    assert(z.size() > 0 && z.size() == pbar.size());

    const size_t sz = z.size();
    // sort z values
    const sizvec sorted_ind = sort_indexes<prec_t>(z);
    // initialize output probability distribution; copy the values because most
    // may be unchanged
    numvec o(pbar);
    // pointer to the smallest (worst case) element
    size_t k = sorted_ind[0];
    // determine how much deviation is actually possible given the provided
    // distribution
    prec_t epsilon = std::min(xi / 2, 1 - pbar[k]);
    // add all the possible weight to the smallest element (structure of the
    // optimal solution)
    o[k] += epsilon;
    // start from the last element
    size_t i = sz - 1;
    // find the upper quantile that corresponds to the epsilon
    while (epsilon > 0) {
        k = sorted_ind[i];
        // compute how much of epsilon remains and can be addressed by the current
        // element
        auto diff = std::min(epsilon, o[k]);
        // adjust the output and epsilon accordingly
        o[k] -= diff;
        epsilon -= diff;
        i--;
    }
    prec_t r = inner_product(o.cbegin(), o.cend(), z.cbegin(), prec_t(0.0));
    return { move(o), r };
}


// Using contraction to solve the inner maximum under a fixed policy
pair<vector<vector<numvec>>, numvec> DLRPG_Innermax_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const vector<vector<prec_t>>& pi_t)
{
    // general constants values
    constexpr double inf = numeric_limits<double>::infinity();

    const vector<vector<numvec>>& P = prob.P;
    const vector<vector<numvec>>& B = prob.B;
    const vector<vector<numvec>>& r = prob.r;
    const numvec& rho = prob.rho;
    const prec_t& gamma = prob.gamma;
    const size_t nStates = prob.P.size();
    const size_t nActions = prob.P[0].size();


    // set zero vector as the initial v_0
    numvec v_0(nStates, 0.0);
    // set temp vector
    numvec v_1(nStates, 0.0);

    // Store the maximum transition kernel
    vector<vector<numvec>> P_t;
    P_t.reserve(nStates);
    for (size_t s = 0; s < nStates; s++)
    {
        vector<numvec> P_t_s;
        P_t_s.reserve(nActions);
        for (size_t a = 0; a < nActions; a++)
        {
            numvec P_t_sa(nStates, 0.0);
            P_t_s.push_back(P_t_sa);
        }
        P_t.push_back(P_t_s);
    }

    while (true)
    {
        for (int s = 0; s < nStates; s++)
        {
            double temp_v = 0.0;
            for (int a = 0; a < nActions; a++)
            {
                numvec temp_c; temp_c.reserve(nStates);
                for (int s_1 = 0; s_1 < nStates; s_1++)
                {
                    //temp_c[s_1] = r[s][a][s_1] + gamma * v_0[s_1];
                    temp_c.push_back(r[s][a][s_1] + gamma * v_0[s_1]);
                }

                double temp_objective_value = 0.0;

                GRBEnv env = GRBEnv(true);
                env.set(GRB_IntParam_OutputFlag, 0);
                //env.set(GRB_IntParam_Threads, 1);
                env.start();
                GRBModel model = GRBModel(env);

                // Define decision variables
                GRBVar* p_sa;
                p_sa = model.addVars(nStates, GRB_CONTINUOUS);
                for (int i = 0; i < nStates; i++)
                {
                    p_sa[i].set(GRB_DoubleAttr_LB, 0.0);
                    p_sa[i].set(GRB_DoubleAttr_UB, 1.0);
                }
                GRBVar* temp_y;
                temp_y = model.addVars(nStates, GRB_CONTINUOUS);
                for (int i = 0; i < nStates; i++)
                {
                    temp_y[i].set(GRB_DoubleAttr_LB, -inf);
                    temp_y[i].set(GRB_DoubleAttr_UB, inf);
                }

                // objective
                GRBLinExpr objective;
                // constraints term
                GRBLinExpr sum_p;
                GRBLinExpr sum_y;
                GRBLinExpr sum_bp;
                // p_sa^T 1 = 1
                for (int s_3 = 0; s_3 < nStates; s_3++)
                {
                    sum_p += p_sa[s_3];
                }
                model.addConstr(sum_p == 1.0);
                // y^T 1 = 1
                for (int s_4 = 0; s_4 < nStates; s_4++)
                {
                    sum_y += temp_y[s_4];
                }
                model.addConstr(sum_y <= kappa[s][a]);

                // y = |p - p_c|
                for (int s_5 = 0; s_5 < nStates; s_5++)
                {
                    model.addConstr(p_sa[s_5] <= temp_y[s_5] + P[s][a][s_5]);
                    model.addConstr(p_sa[s_5] >= P[s][a][s_5] - temp_y[s_5]);
                }

                // branch location constraint

                for (int s_6 = 0; s_6 < nStates; s_6++)
                {
                    sum_bp += p_sa[s_6] * B[s][a][s_6];
                }
                model.addConstr(sum_bp == 0.0);
                // set objective
                for (int s_7 = 0; s_7 < nStates; s_7++) {
                    objective += p_sa[s_7] * temp_c[s_7];
                }
                model.setObjective(objective, GRB_MAXIMIZE);
                //model.setObjective(objective, GRB_MINIMIZE);
                // run optimization
                model.optimize();
                temp_objective_value = model.get(GRB_DoubleAttr_ObjVal);
                temp_v += pi_t[s][a] * temp_objective_value;
                for (int s_8 = 0; s_8 < nStates; s_8++)
                {
                    P_t[s][a][s_8] = p_sa[s_8].get(GRB_DoubleAttr_X);
                }
                //p_sa.get(GRB_DoubleAttr_X)

            }
            v_1[s] = temp_v;
        }
        double gap_2 = 0;
        for (size_t s_9 = 0; s_9 < nStates; s_9++)
        {
            gap_2 += (v_1[s_9] - v_0[s_9]) * (v_1[s_9] - v_0[s_9]);
        }
        double gap = sqrt(gap_2);
        if (gap <= 1e-8)
        {
            return { P_t, v_0 };
            break;
        }
        v_0 = v_1;
    }
}


// Using RAAM inner method to solve the inner maximum under a fixed policy
pair<vector<vector<numvec>>, numvec> DLRPG_RAAM_Innermax_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const vector<vector<prec_t>>& pi_t)
{
    // general constants values
    constexpr double inf = numeric_limits<double>::infinity();

    const vector<vector<numvec>>& P = prob.P;
    const vector<vector<numvec>>& B = prob.B;
    const vector<vector<numvec>>& r = prob.r;
    const numvec& rho = prob.rho;
    const prec_t& gamma = prob.gamma;
    const size_t nStates = prob.P.size();
    const size_t nActions = prob.P[0].size();


    // set zero vector as the initial v_0
    numvec v_0(nStates, 0.0);
    // set temp vector
    numvec v_1(nStates, 0.0);

    // Store the maximum transition kernel
    vector<vector<numvec>> P_t;
    P_t.reserve(nStates);
    for (size_t s = 0; s < nStates; s++)
    {
        vector<numvec> P_t_s;
        P_t_s.reserve(nActions);
        for (size_t a = 0; a < nActions; a++)
        {
            numvec P_t_sa(nStates, 0.0);
            P_t_s.push_back(P_t_sa);
        }
        P_t.push_back(P_t_s);
    }

    while (true)
    {
        for (int s = 0; s < nStates; s++)
        {
            double temp_v = 0.0;
            for (int a = 0; a < nActions; a++)
            {
                numvec temp_c; temp_c.reserve(nStates);
                for (int s_1 = 0; s_1 < nStates; s_1++)
                {
                    //temp_c.push_back( r[s][a][s_1] + gamma * v_0[s_1]);
                    temp_c.push_back(-r[s][a][s_1] - gamma * v_0[s_1]);
                }

                auto [P_sa, temp_objective_value] = worstcase_l1(temp_c, P[s][a],
                    kappa[s][a]);


                temp_v += pi_t[s][a] * -1 * temp_objective_value;
                for (int s_8 = 0; s_8 < nStates; s_8++)
                {
                    //P_t[s][a][s_8] = p_sa[s_8].get(GRB_DoubleAttr_X);
                    P_t[s][a][s_8] = P_sa[s_8];
                }
                //p_sa.get(GRB_DoubleAttr_X)

            }
            v_1[s] = temp_v;
        }
        double gap_2 = 0;
        for (size_t s_9 = 0; s_9 < nStates; s_9++)
        {
            gap_2 += (v_1[s_9] - v_0[s_9]) * (v_1[s_9] - v_0[s_9]);
        }
        double gap = sqrt(gap_2);
        if (gap <= 1e-8)
        {
            return { P_t, v_0 };
            break;
        }
        v_0 = v_1;
    }
}


//Compute the value function for fixed pi and p
numvec MDP_value(const RMDPs_SA& prob, const vector<vector<prec_t>>& pi_now, const vector<vector<numvec>>& P_now)
{
    MatrixXd I = MatrixXd::Identity(prob.nStates, prob.nStates);
    numvec c; c.reserve(prob.nStates);
    for (int s = 0; s < prob.nStates; s++)
    {
        double mid1 = 0;
        for (int a = 0; a < prob.nActions; a++)
        {
            double mid2 = 0;
            for (int s_prime = 0; s_prime < prob.nStates; s_prime++)
            {
                mid2 += P_now[s][a][s_prime] * prob.r[s][a][s_prime];
            }
            mid1 += pi_now[s][a] * mid2;
        }
        c.push_back(mid1);
    }
    VectorXd R_pi(prob.nStates);
    for (int s = 0; s < prob.nStates; s++)
    {
        R_pi(s) = c[s];
    }
    vector<vector<prec_t>> P_pi; P_pi.reserve(prob.nStates);
    for (int s = 0; s < prob.nStates; s++) {
        numvec P_pi_s; P_pi_s.reserve(prob.nStates);
        for (int s_prime = 0; s_prime < prob.nStates; s_prime++) {
            double temp = 0;
            for (int a = 0; a < prob.nActions; a++)
            {
                temp += P_now[s][a][s_prime] * pi_now[s][a];
            }
            P_pi_s.push_back(temp);
        }
        P_pi.push_back(P_pi_s);
    }
    MatrixXd mat_P_pi(prob.nStates, prob.nStates);
    for (int s = 0; s < prob.nStates; s++)
    {
        for (int s_prime = 0; s_prime < prob.nStates; s_prime++)
        {
            mat_P_pi(s, s_prime) = P_pi[s][s_prime];
        }
    }
    MatrixXd mat_temp1(prob.nStates, prob.nStates);
    //MatrixXd mat_temp2(prob.nStates, prob.nStates);
    mat_temp1 = I - prob.gamma * mat_P_pi;
    VectorXd x = mat_temp1.colPivHouseholderQr().solve(R_pi);
    numvec V(prob.nStates, 0.0);
    for (int s = 0; s < prob.nStates; s++)
    {
        V[s] = x(s);
    }
    return V;
}


// Compute the partial gradient matrix for transition kernel P
vector<vector<numvec>> MDP_grad_P(const RMDPs_SA& prob, const vector<vector<prec_t>>& pi_now, const vector<vector<numvec>>& P_now, const numvec& eta, const numvec& v_now)
{
    // Store the gradient of transition kernel
    vector<vector<numvec>> G_P;
    G_P.reserve(prob.nStates);
    for (int s = 0; s < prob.nStates; s++)
    {
        vector<numvec> G_P_s;
        G_P_s.reserve(prob.nActions);
        for (int a = 0; a < prob.nActions; a++)
        {
            numvec G_P_sa;
            G_P_sa.reserve(prob.nStates);
            double temp_c;
            double grad;
            for (int s1 = 0; s1 < prob.nStates; s1++)
            {
                temp_c = prob.r[s][a][s1] + prob.gamma * v_now[s1];
                grad = (1 / (1 - prob.gamma)) * eta[s] * pi_now[s][a] * temp_c;
                G_P_sa.push_back(grad);
            }
            G_P_s.push_back(G_P_sa);
        }
        G_P.push_back(G_P_s);
    }
    return G_P;
}


// Using projected gradient method to solve the inner maximum under a fixed policy
pair<vector<vector<numvec>>, numvec> DLRPG_PGD_Innermax_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const vector<vector<prec_t>>& pi_t, const vector<vector<numvec>>& P_ini, const double& step)
{
    constexpr double inf = numeric_limits<double>::infinity();

    const vector<vector<numvec>>& P = prob.P;
    const vector<vector<numvec>>& B = prob.B;
    const vector<vector<numvec>>& r = prob.r;
    const numvec& rho = prob.rho;
    const prec_t& gamma = prob.gamma;
    const size_t nStates = prob.P.size();
    const size_t nActions = prob.P[0].size();

    numvec v_0;
    numvec v_1(nStates, 0.0);
    int t = 0;
    vector<vector<numvec>> P_0 = P_ini;

    // Store the maximum transition kernel
    vector<vector<numvec>> P_t;
    P_t.reserve(nStates);
    for (size_t s = 0; s < nStates; s++)
    {
        vector<numvec> P_t_s;
        P_t_s.reserve(nActions);
        for (size_t a = 0; a < nActions; a++)
        {
            numvec P_t_sa(nStates, 0.0);
            P_t_s.push_back(P_t_sa);
        }
        P_t.push_back(P_t_s);
    }
    while (true)
    {
        t += 1;
        v_0 = MDP_value(prob, pi_t, P_0);
        numvec eta_now = MDP_Occu(prob, P_0, pi_t);
        vector<vector<numvec>> MDP_G_P = MDP_grad_P(prob, pi_t, P_0, eta_now, v_0);
        for (int s = 0; s < prob.nStates; s++)
        {
            for (int a = 0; a < prob.nActions; a++)
            {
                //double temp_objective_value;
                GRBEnv env = GRBEnv(true);
                env.set(GRB_IntParam_OutputFlag, 0);
                //env.set(GRB_IntParam_Threads, 1);
                env.start();
                GRBModel model = GRBModel(env);

                // Define decision variables
                GRBVar* p_sa;
                p_sa = model.addVars(nStates, GRB_CONTINUOUS);
                for (int i = 0; i < nStates; i++)
                {
                    p_sa[i].set(GRB_DoubleAttr_LB, 0.0);
                    p_sa[i].set(GRB_DoubleAttr_UB, 1.0);
                }
                GRBVar* temp_y;
                temp_y = model.addVars(nStates, GRB_CONTINUOUS);
                for (int i = 0; i < nStates; i++)
                {
                    temp_y[i].set(GRB_DoubleAttr_LB, -inf);
                    temp_y[i].set(GRB_DoubleAttr_UB, inf);
                }

                // objective
                GRBQuadExpr objective;
                // constraints term
                GRBLinExpr sum_p;
                GRBLinExpr sum_y;
                GRBLinExpr sum_bp;
                // p_sa^T 1 = 1
                for (int s_3 = 0; s_3 < nStates; s_3++)
                {
                    sum_p += p_sa[s_3];
                }
                model.addConstr(sum_p == 1.0);
                // y^T 1 <= kappa
                for (int s_4 = 0; s_4 < nStates; s_4++)
                {
                    sum_y += temp_y[s_4];
                }
                model.addConstr(sum_y <= kappa[s][a]);

                // y = |p - p_c|
                for (int s_5 = 0; s_5 < nStates; s_5++)
                {
                    model.addConstr(p_sa[s_5] <= temp_y[s_5] + P[s][a][s_5]);
                    model.addConstr(p_sa[s_5] >= P[s][a][s_5] - temp_y[s_5]);
                }

                // branch location constraint

                for (int s_6 = 0; s_6 < nStates; s_6++)
                {
                    sum_bp += p_sa[s_6] * B[s][a][s_6];
                }
                model.addConstr(sum_bp == 0.0);
                // set objective
                for (int s_7 = 0; s_7 < nStates; s_7++) {
                    objective += (p_sa[s_7] - P_t[s][a][s_7]) * (p_sa[s_7] - P_t[s][a][s_7]);
                    objective -= 2 * step * p_sa[s_7] * MDP_G_P[s][a][s_7];
                }
                model.setObjective(objective, GRB_MINIMIZE);
                // run optimization
                model.optimize();
                //temp_objective_value = model.get(GRB_DoubleAttr_ObjVal);
                for (int s_8 = 0; s_8 < nStates; s_8++)
                {
                    P_t[s][a][s_8] = p_sa[s_8].get(GRB_DoubleAttr_X);
                }
            }
        }
        double gap = 0;
        for (size_t s_9 = 0; s_9 < nStates; s_9++)
        {
            gap += (v_0[s_9] - v_1[s_9]) * (v_0[s_9] - v_1[s_9]);
        }
        double gap1 = sqrt(gap);
        if (gap1 <= 1e-3)
        {
            return { P_t, v_0 };
            break;
        }
        P_0 = P_t;
        v_1 = v_0;
    }
}


// Compute the partial gradient matrix for policy pi
vector<vector<prec_t>> MDP_grad_pi(const RMDPs_SA& prob, const vector<vector<numvec>>& P_now, const numvec& eta, const numvec& v_now)
{
    vector<vector<prec_t>> G_pi; G_pi.reserve(prob.nStates);
    for (int s = 0; s < prob.nStates; s++)
    {
        numvec G_pi_s; G_pi_s.reserve(prob.nActions);
        for (int a = 0; a < prob.nActions; a++)
        {
            numvec temp_c; temp_c.reserve(prob.nStates);
            for (int s_1 = 0; s_1 < prob.nStates; s_1++)
            {
                //temp_c[s_1] = r[s][a][s_1] + gamma * v_0[s_1];
                temp_c.push_back(prob.r[s][a][s_1] + prob.gamma * v_now[s_1]);
            }
            double q_sa = 0;
            for (int i = 0; i < prob.nStates; i++)
            {
                q_sa += temp_c[i] * P_now[s][a][i];
            }
            double temp_G_sa = (1 / (1 - prob.gamma)) * eta[s] * q_sa;
            G_pi_s.push_back(temp_G_sa);
        }
        G_pi.push_back(G_pi_s);
    }
    return G_pi;
}


// Update pi by projected gradient descent (gorubi)
vector<vector<prec_t>> DLRPG_outermin_SA(const RMDPs_SA& prob, const vector<vector<prec_t>>& pi_old, const double& step, const vector<vector<prec_t>>& G_now)
{
    constexpr double inf = numeric_limits<double>::infinity();
    const size_t nStates = prob.P.size();
    const size_t nActions = prob.P[0].size();

    vector<vector<prec_t>> pi_new; pi_new.reserve(nStates);
    for (int s_0 = 0; s_0 < nStates; s_0++)
    {
        numvec pi_new_s(nActions, 0.0);
        pi_new.push_back(pi_new_s);
    }

    for (int s = 0; s < nStates; s++)
    {
        GRBEnv env = GRBEnv(true);
        env.set(GRB_IntParam_OutputFlag, 0);
        //env.set(GRB_IntParam_Threads, 1);
        env.start();
        GRBModel model = GRBModel(env);

        // Define decision variables
        GRBVar* pi_s;
        pi_s = model.addVars(nActions, GRB_CONTINUOUS);
        for (int i = 0; i < nActions; i++)
        {
            pi_s[i].set(GRB_DoubleAttr_LB, 0.0);
            pi_s[i].set(GRB_DoubleAttr_UB, 1.0);
        }
        // objective
        GRBQuadExpr objective;
        // constraints term
        GRBLinExpr sum_pi_s;
        for (int a_0 = 0; a_0 < nActions; a_0++)
        {
            sum_pi_s += pi_s[a_0];
        }
        model.addConstr(sum_pi_s == 1.0);
        // set objective

        for (int a_1 = 0; a_1 < nActions; a_1++) {
            objective += (pi_s[a_1] - pi_old[s][a_1]) * (pi_s[a_1] - pi_old[s][a_1]);
            objective += 2 * step * pi_s[a_1] * G_now[s][a_1];
        }

        model.setObjective(objective, GRB_MINIMIZE);
        // run optimization
        model.optimize();

        for (int a_2 = 0; a_2 < nActions; a_2++)
        {
            pi_new[s][a_2] = pi_s[a_2].get(GRB_DoubleAttr_X);
        }
    }
    return pi_new;
}

// Compute l2-norm for vector library
double L2_Gap(const numvec& v1, const numvec& v2)
{
    int S = v1.size();
    numvec v_gap;
    for (int i_1 = 0; i_1 < S; i_1++)
    {
        v_gap.push_back(v1[i_1] - v2[i_1]);
    }
    double sum1 = 0.0;
    for (int i_2 = 0; i_2 < S; i_2++)
    {
        sum1 += v_gap[i_2] * v_gap[i_2];
    }
    double gap = sqrt(sum1);
    return gap;
}
double inner_product(const numvec& v1, const numvec& v2)
{
    int S = v1.size();
    double x = 0.0;
    for (int i = 0; i < S; i++)
    {
        x += v1[i] * v2[i];
    }
    return x;
}


// DLRPG with Gorubi
pair<vector<vector<prec_t>>, numvec> DLRPG_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const double& step, const numvec& v_opt)
{
    vector<vector<prec_t>> pi_old = Rand_Policy(prob);
    int n = 0;
    while (true)
    {
        n++;
        cout << n << endl;
        auto [P_inner, V_inner] = DLRPG_Innermax_SA(prob, kappa, pi_old);
        numvec eta_now = MDP_Occu(prob, P_inner, pi_old);
        vector<vector<prec_t>> MDP_PiG = MDP_grad_pi(prob, P_inner, eta_now, V_inner);
        vector<vector<prec_t>> pi_new = DLRPG_outermin_SA(prob, pi_old, step, MDP_PiG);
        double gap = L2_Gap(V_inner, v_opt);
        if (gap <= 1e-8)
        {
            cout << "counts: " << n << endl;
            return { pi_new, V_inner };
            break;
        }

        for (int s = 0; s < prob.nStates; s++)
        {
            for (int a = 0; a < prob.nActions; a++)
            {
                pi_old[s][a] = pi_new[s][a];
            }
        }

    }
}


// DLRPG with RAAM
pair<vector<vector<prec_t>>, numvec> DLRPG_RAAM_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const double& step, const numvec& v_opt)
{
    vector<vector<prec_t>> pi_old = Rand_Policy(prob);
    int n = 0;
    double gap;
    while (true)
    {
        n++;
        cout << n << endl;
        auto [P_inner, V_inner] = DLRPG_RAAM_Innermax_SA(prob, kappa, pi_old);
        numvec eta_now = MDP_Occu(prob, P_inner, pi_old);
        vector<vector<prec_t>> MDP_PiG = MDP_grad_pi(prob, P_inner, eta_now, V_inner);
        vector<vector<prec_t>> pi_new = DLRPG_outermin_SA(prob, pi_old, step, MDP_PiG);
        gap = L2_Gap(V_inner, v_opt);
        //cout << gap << endl;
        if (gap <= 1e-8)
        {
            cout << "counts: " << n << endl;
            return { pi_new, V_inner };
            break;
        }
        for (int s = 0; s < prob.nStates; s++)
        {
            for (int a = 0; a < prob.nActions; a++)
            {
                pi_old[s][a] = pi_new[s][a];
            }
        }

    }
}


// DLRPG with PGD by Gorubi
tuple<vector<vector<prec_t>>, numvec, numvec> DLRPG_PGD_SA(const RMDPs_SA& prob, const vector<numvec>& kappa, const double& out_step, const double& in_step, const int& ite_time)
{
    vector<vector<prec_t>> pi_old = Rand_Policy(prob);
    //numvec v_1(prob.nStates, 0.0);
    int n = 0;
    numvec J; J.reserve(ite_time);
    while (true)
    {
        n++;
        auto [P_inner, V_inner] = DLRPG_PGD_Innermax_SA(prob, kappa, pi_old, prob.P, in_step);
        numvec eta_now = MDP_Occu(prob, P_inner, pi_old);
        vector<vector<prec_t>> MDP_PiG = MDP_grad_pi(prob, P_inner, eta_now, V_inner);
        vector<vector<prec_t>> pi_new = DLRPG_outermin_SA(prob, pi_old, out_step, MDP_PiG);
        double J_ite = inner_product(V_inner, prob.rho);
        J.push_back(J_ite);
        if (n == ite_time)
        {
            return { pi_new, V_inner, J };
            break;
        }

        //v_1 = V_inner;
        for (int s = 0; s < prob.nStates; s++)
        {
            for (int a = 0; a < prob.nActions; a++)
            {
                pi_old[s][a] = pi_new[s][a];
            }
        }

    }
}



void Sample_sa(const int& Garnet_Snum, const int& Garnet_Anum, const int& Garnet_Bnum, const int& itenum, const double& out_step, const double& in_step, const int& sample_num)
{
    vector<numvec> DM_RVI; DM_RVI.reserve(sample_num);
    vector<numvec> DM_DLRPG; DM_DLRPG.reserve(sample_num);
    for (int i = 0; i < sample_num; i++)
    {
        cout << i << endl;
        RMDPs_SA MDP = Garnet_SARMDPs(Garnet_Snum, Garnet_Anum, Garnet_Bnum);
        // Give a close initial value
        vector<vector<double>> pi_ini = Rand_Policy(MDP);
        vector<numvec> Kappa = Rand_tolerance(MDP);
        numvec V_ini = MDP_value(MDP, pi_ini, MDP.P);
        auto [Pi_RVI, V_RVI, J_RVI] = RVI_SA(MDP, Kappa, itenum, V_ini);
        auto [Pi_PGD, V_DLRPG_PGD, J_DLRPG] = DLRPG_PGD_SA(MDP, Kappa, out_step, in_step, itenum);
        DM_RVI.push_back(J_RVI);
        DM_DLRPG.push_back(J_DLRPG);
    }

    string filename = "F:/Cityu/Research/Robust Optimization/Robust MDP/Policy gradient/AISTATS2023PaperPack/Data and result/sa-rec 50samples for RVI S=5 ver2.csv";
    ofstream ofs(filename, ofstream::out);


    for (int i = 0; i < sample_num; i++)
    {
        for (int j = 0; j < itenum; j++)
        {
            ofs << DM_RVI[i][j] << ",";
        }
        ofs << "\n";
    }

    ofs.close();

    cout << "Finish the RVI yet" << endl;

    string filename1 = "F:/Cityu/Research/Robust Optimization/Robust MDP/Policy gradient/AISTATS2023PaperPack/Data and result/sa-rec 50samples for DLRPG S=5 ver2.csv";
    ofstream ofs1(filename1, ofstream::out);


    for (int i = 0; i < sample_num; i++)
    {
        for (int j = 0; j < itenum; j++)
        {
            ofs1 << DM_DLRPG[i][j] << ",";
        }
        ofs1 << "\n";
    }

    ofs1.close();

}
