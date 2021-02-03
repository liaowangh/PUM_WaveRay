#include <fstream>
#include <string>

#include "utils.h"
#include "../pum_wave_ray/HE_FEM.h"
#include "../pum_wave_ray/PUM_WaveRay.h"

Scalar LocalIntegral(const lf::mesh::Entity& e, int quad_degree, const FHandle_t& f) {
    auto qr = lf::quad::make_QuadRule(e.RefEl(), quad_degree);
    auto global_points = e.Geometry()->Global(qr.Points());
    auto weights_ie = (qr.Weights().cwiseProduct(e.Geometry()->IntegrationElement(qr.Points()))).eval();

    Scalar temp = 0.0;
    for (Eigen::Index i = 0; i < qr.NumPoints(); ++i) {
        temp += weights_ie(i) * f(global_points.col(i));
    }
    return temp;
}

void print_save_error(std::vector<int>& N, std::vector<std::vector<double>>& data, 
    std::vector<std::string>& err_str, const std::string& sol_name, 
    const std::string& output_folder) {
    
    std::cout << sol_name << std::endl;
    //Tabular output of the data
    std::cout << std::left << std::setw(10) << "N";
    for(int i = 0; i < err_str.size(); ++i){
        std::cout << std::setw(20) << err_str[i];
    }
    std::cout << std::endl;
    for(int l = 0; l < N.size(); ++l) {
        std::cout << std::left << std::setw(10) << N[l];
        for(int i = 0; i < data.size(); ++i) {
            std::cout << std::setw(20) << data[i][l];
        }
        std::cout << std::endl;
    }

    // write the result to the file
    std::string output_file = output_folder + sol_name + ".txt";
    std::ofstream out(output_file);

    out << "N";
    for(int i = 0; i < err_str.size(); ++i) {
        out << " " << err_str[i];
    }
    out << std::endl;

    for(int l = 0; l < N.size(); ++l) {
        out << N[l];
        for(int i = 0; i < data.size(); ++i) {
            out << " " << data[i][l];
        }
        out << std::endl;
    } 
}

void test_solve(HE_FEM& he_fem, const std::string& sol_name, 
    const std::string& output_folder, size_type L, const FHandle_t& u, 
    const FunGradient_t& grad_u) {
    std::vector<int> ndofs;
    std::vector<double> L2err, H1serr, H1err;
    
    for(size_type level = 0; level <= L; ++level) {
        auto fe_sol = he_fem.solve(level);
        
        double l2_err = he_fem.L2_Err(level, fe_sol, u);
        double h1_serr = he_fem.H1_semiErr(level, fe_sol, grad_u);
        double h1_err = std::sqrt(l2_err*l2_err + h1_serr*h1_serr);
        
        ndofs.push_back(fe_sol.size());
        L2err.push_back(l2_err);
        H1serr.push_back(h1_serr);
        H1err.push_back(h1_err);
    }
    
    std::vector<std::vector<double>> err_data{L2err, H1err, H1serr};
    std::vector<std::string> err_str{"L2_err", "H1_err", "H1_serr"};
    print_save_error(ndofs, err_data, err_str, sol_name, output_folder);
}

void Gaussian_Seidel(Mat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu){
    // u: initial value; mu: number of iterations
    int N = A.rows();
    for(int i = 0; i < mu; ++i){
        for(int t = 0; t < stride; ++t) {
            // stride/direction
            for(int k = 0; k < N / stride; ++k) {
                int j = k * stride + t;
                Scalar tmp = (A.row(j) * u)(0,0);
                u(j) = (phi(j) - tmp + u(j) * A(j,j)) / A(j,j);
            }
        }
    }
}

void Gaussian_Seidel(Mat_t& A, Vec_t& phi, Vec_t& u, Vec_t& sol, int stride){
    // u: initial value;
    int N = A.rows();
    std::cout << std::left << std::setw(10) << "Iteration"
        << std::setw(20) << "Err_norm" << std::endl;

    int cnt = 0;
    while(true){
        cnt++;
        for(int t = 0; t < stride; ++t) {
            // stride/direction
            for(int k = 0; k < N / stride; ++k) {
                int j = k * stride + t;
                Scalar tmp = (A.row(j) * u)(0,0);
                u(j) = (phi(j) - tmp + u(j) * A(j,j)) / A(j,j);
            }
        }
        if(cnt % 20 == 0) {
            std::cout << std::left << std::setw(10) << cnt
                << std::setw(20) << (sol - u).norm() / sol.norm() << std::endl;
        }
        
        if((sol - u).norm() < 0.001){
            break;
        }
        if(cnt > 500) {
            std::cout << "Gaussian Seidel iteration doesn't converge after "
                      << cnt << " iterations." << std::endl; 
            break;
        }
    }
}

std::pair<Vec_t, Scalar> power_GS(Mat_t& A, int stride) {
    /* Compute the Eigen value of the GS operator */
    Mat_t L = Mat_t(A.triangularView<Eigen::Lower>());
    Mat_t U = A - L;
    Mat_t GS_op = L.colPivHouseholderQr().solve(-U);
    Vec_t eivals = GS_op.eigenvalues();

    Scalar domainant_eival = eivals(0);
    for(int i = 1; i < eivals.size(); ++i) {
        if(std::abs(eivals(i)) > std::abs(domainant_eival)) {
            domainant_eival = eivals(i);
        }
    }
    std::cout << "Domainant eigenvalue: " << domainant_eival << std::endl;
    std::cout << "Absolute value: " << std::abs(domainant_eival) << std::endl;
    /**********************************************/

    double tol = 1e-3;
    int N = A.rows();
    Vec_t u = Vec_t::Random(N);

    u.normalize();    
    Scalar lambda;
    double normAest = A.operatorNorm();
    int cnt = 0;

    std::cout << "normAset: " << normAest << std::endl;
    std::cout << std::left << std::setw(10) << "Iteration"
        << std::setw(20) << "residual_norm" << std::endl;
    while(1){
        cnt++;
        Vec_t old_u = u;
        for(int t = 0; t < stride; ++t) {
            for(int k = 0; k < N / stride; ++k) {
                int j = k * stride + t;
                Scalar tmp = (A.row(j) * u)(0,0);
                u(j) = (u(j) * A(j,j) - tmp) / A(j,j);
            }
        }
        // now u should be GS_op * old_u
        lambda = old_u.dot(u); // Rayleigh quotient
        // compute the residual and check vs tolerance
        auto r = u - lambda * old_u;

        if(cnt % 20 == 0){
            std::cout << std::left << std::setw(10) << cnt
                << std::setw(20) << r.norm() << std::endl;
        }

        u.normalize();
        
        if(r.norm() < tol) {
            std::cout << "Power iteration converges after " << cnt 
                << " iterations." << std::endl;
            break;
        }
        if(cnt > 500) {
            std::cout << "Power iteration doesn't converge after " << cnt 
                << " iterations." << std::endl; 
            break;
        }
    }
    std::cout << "Number of iterations: " << cnt << std::endl;
    std::cout << "Domainant eigenvalue by power iteration: " << lambda << std::endl;
    return std::make_pair(u, lambda);
}

/*
 * Perform v-cycle, finer grid transfer the residual to the coarser grid, 
 * in which the residual equation is solved, and then the error is transfered back to finer grid.
 * 
 * u: initial value.
 * f: r.h.s vector
 * Op: container storing all the operators.
 * I: transfer operators, I[i]: Mesh_i -> Mesh_{i+1}
 * stride: stride in Gaussian Seidel relaxation
 * mu1, mu2: pre and post relaxation times
 */
void v_cycle(Vec_t& u, Vec_t& f, std::vector<Mat_t>& Op, std::vector<Mat_t>& I, 
    std::vector<int>& stride, size_type mu1, size_type mu2) {

    Vec_t old_u = u;

    int L = I.size();
    LF_ASSERT_MSG(Op.size() == L + 1 && stride.size() == L + 1, 
        "#{transfer operator} should be #{Operator} - 1");
    
    std::vector<int> op_size(L+1);
    for(int i = 0; i <= L; ++i) {
        op_size[i] = Op[i].rows();
    }

    for(int i = 0; i < L; ++i) {
        LF_ASSERT_MSG(I[i].rows() == op_size[i+1] && I[i].cols() == op_size[i],
            "transfer operator size does not mathch grid operator size.");
    }

    std::vector<Vec_t> initial(L + 1), rhs_vec(L + 1);

    initial[L] = u;
    rhs_vec[L] = f;
    // initial guess on coarser mesh are all zero
    for(int i = 0; i < L; ++i) {
        initial[i] = Vec_t::Zero(op_size[i]);
    }

    // std::cout << "Finer to coarser" << std::endl;
    for(int i = L; i > 0; --i) {
        Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], stride[i], mu1);
        rhs_vec[i-1] = I[i-1].transpose() * (rhs_vec[i] - Op[i] * initial[i]);
    }
    // std::cout << "solve on coarest mesh" << std::endl;
    initial[0] = Op[0].colPivHouseholderQr().solve(rhs_vec[0]);

    // std::cout << "Coarser to finer" << std::endl;
    for(int i = 1; i <= L; ++i) {
        initial[i] += I[i-1] * initial[i-1];
        Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], stride[i], mu2);
    }
    u = initial[L];

    // std::cout << "After v cycle, ||u-u_old||=" << (old_u - u).norm() << std::endl;
}