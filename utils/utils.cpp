#include <fstream>
#include <random>
#include <string>

#include "utils.h"
#include "HE_solution.h"
#include "../Pum_WaveRay/HE_FEM.h"
#include "../Pum_WaveRay/PUM_WaveRay.h"

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

Scalar integrate(std::shared_ptr<lf::mesh::Mesh> mesh, const FHandle_t& f, int degree){
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});

    Scalar res = 0.0;
    for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
        res += LocalIntegral(*cell, degree, f);
    }
    return res;
}

double L2_norm(std::shared_ptr<lf::mesh::Mesh> mesh, const FHandle_t& f, int degree=20) {
    auto f_square = [&f](const coordinate_t& x) {
        return std::abs(f(x) * f(x));
    };
    auto res = integrate(mesh, f_square, degree);
    return std::sqrt(std::abs(res));
}

/*
 * Do the linear regression, return {k, b} such that
 *  y_i \approx k*x_i + b
 */
std::vector<double> linearFit(const std::vector<double> x, const std::vector<double> y) {
    Eigen::Matrix2d XTX;
    Eigen::Vector2d XTY;
    XTX.setZero();
    XTY.setZero();
    for(int i = 0; i < x.size(); ++i) {
        XTX(0,0) += x[i] * x[i];
        XTX(0,1) += x[i];
        XTY(0) += x[i] * y[i];
        XTY(1) += y[i];
    }
    XTX(1,0) = XTX(0,1);
    XTX(1,1) = x.size();

    auto tmp = XTX.colPivHouseholderQr().solve(XTY);
    return {tmp(0), tmp(1)};
}

void tabular_output(std::vector<std::vector<double>>& data, 
    std::vector<std::string>& data_label, const std::string& sol_name, 
    const std::string& output_folder, bool save) {
    
    std::cout << sol_name << std::endl;
    //Tabular output of the data
    // std::cout << std::left << std::setw(10) << data_label[i];
    std::cout << std::left;
    for(int i = 0; i < data_label.size(); ++i){
        std::cout << std::setw(10) << data_label[i];
    }
    std::cout << std::endl;
    // std::cout << std::left << std::scientific << std::setprecision(1);
    for(int l = 0; l < data[0].size(); ++l) {
        std::cout << l << " & ";
        for(int i = 0; i < data.size(); ++i) {
            std::cout << std::setw(10) << data[i][l];
            if(i == data.size() - 1) {
                std::cout << " \\\\";
            } else {
                std::cout << " & ";
            }
        }
        std::cout << std::endl;
    }

    // write the result to the file
    if(save){
        std::string output_file = output_folder + sol_name + ".txt";
        std::ofstream out(output_file);

        out << data_label[0];
        for(int i = 1; i < data_label.size(); ++i) {
            out << " " << data_label[i];
        }
        out << std::endl;
        out << std::scientific << std::setprecision(1);
        for(int l = 0; l < data[0].size(); ++l) {
            out << l << " & ";
            for(int i = 0; i < data.size(); ++i) {
                if(i == data.size() - 1) {
                    out << data[i][l] << " \\\\";
                } else {
                    out << data[i][l] << " & ";
                }
            }
            out << std::endl;
        } 
    }
}

void test_solve(HE_FEM& he_fem, const std::string& sol_name, 
    const std::string& output_folder, int L, const FHandle_t& u, 
    const FunGradient_t& grad_u) {
    // std::vector<int> ndofs;
    std::vector<double> mesh_width = he_fem.mesh_width();
    std::vector<double> L2err, H1serr, H1err;
    
    for(int level = 0; level <= L; ++level) {
        auto fe_sol = he_fem.solve(level);
        
        double l2_err = he_fem.L2_Err(level, fe_sol, u);
        double h1_serr = he_fem.H1_semiErr(level, fe_sol, grad_u);
        double h1_err = std::sqrt(l2_err*l2_err + h1_serr*h1_serr);
        
        // ndofs.push_back(fe_sol.size());
        L2err.push_back(l2_err);
        H1serr.push_back(h1_serr);
        H1err.push_back(h1_err);
    }
    
    std::vector<std::vector<double>> err_data{mesh_width, L2err, H1err, H1serr};
    std::vector<std::string> data_label{"h", "L2_err", "H1_err", "H1_serr"};
    print_save_error(err_data, data_label, sol_name, output_folder);
}

// void Gaussian_Seidel(const SpMat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu){
//     // u: initial value; mu: number of iterations
//     int N = A.rows();
//     for(int i = 0; i < mu; ++i){
//         for(int t = 0; t < stride; ++t) {
//             // stride/direction
//             for(int k = 0; k < N / stride; ++k) {
//                 int j = k * stride + t;
//                 Scalar tmp = (A.row(j) * u)(0,0);
//                 // u(j) = (phi(j) - tmp + u(j) * A(j,j)) / A(j,j);
//                 Scalar Ajj = A.coeffRef(j,j);
//                 u(j) = (phi(j) - tmp + u(j) * Ajj) / Ajj;
//             }
//         }
//     }
// }

void Gaussian_Seidel(SpMat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu){
    // u: initial value; mu: number of iterations
    int N = A.rows();
    for(int i = 0; i < mu; ++i){
        for(int t = 0; t < stride; ++t) {
            // stride/direction
            for(int k = 0; k < N / stride; ++k) {
                int j = k * stride + t;
                Scalar tmp = (A.row(j) * u)(0,0);
                // u(j) = (phi(j) - tmp + u(j) * A(j,j)) / A(j,j);
                Scalar Ajj = A.coeffRef(j,j);
                u(j) = (phi(j) - tmp + u(j) * Ajj) / Ajj;
            }
        }
    }
}

void Gaussian_Seidel(SpMat_t& A, Vec_t& phi, Vec_t& u, Vec_t& sol, int stride){
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
                Scalar Ajj = A.coeffRef(j,j);
                u(j) = (phi(j) - tmp + u(j) * Ajj) / Ajj;
                // u(j) = (phi(j) - tmp + u(j) * A(j,j)) / A(j,j);
            }
        }
        double err = (sol - u).norm() / sol.norm();
        if(cnt % 20 == 0) {
            std::cout << std::left << std::setw(10) << cnt
                << std::setw(20) << err << std::endl;
        }
        if(err < 0.01){
            std::cout << "Gauss Seidel iteration converges after " << cnt << " iterations." << std::endl;
            break;
        }
        if(cnt >= 500) {
            std::cout << "Gauss Seidel iteration doesn't converge after "
                      << cnt << " iterations." << std::endl; 
            break;
        }
    }
}

/*
 * N = stride (number of plan waves), n = u.size() / N (number of nodes)
 * u is divided into blocks
 *  1. {u[i*N],...,u[(i+1)*N-1]}, i = 0, 1, ..., n (divid according to nodes)
 *  2. {u[t], u[N+t], u[2N+t], ... , u[(n-1)N+t]}, t = 0, 1, ... , N-1 (divid according to waves)
 */
// void block_GS(const SpMat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu){
//     LF_ASSERT_MSG(phi.size() % stride == 0, 
//         "the size of unknows should divide stride!");
//     if(stride == 1) {
//         Gaussian_Seidel(A, phi, u, stride, mu);
//         return;
//     }
//     int N = stride;
//     int n = u.size() / N;
//     for(int nu = 0; nu < mu; ++nu) {
//         for(int i = 0; i < n; ++i) {
//             Mat_t Ai = A.block(i*N, i*N, N, N);
//             Vec_t rhs_i = phi.segment(i*N, N) - A.block(i*N, 0, N, N*n) * u 
//                 + Ai * u.segment(i*N, N);
//             u.segment(i*N, N) = Ai.colPivHouseholderQr().solve(rhs_i);
//         }
//     }
// }

void block_GS(SpMat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu){
    LF_ASSERT_MSG(phi.size() % stride == 0, 
        "the size of unknows should divide stride!");
    if(stride == 1) {
        Gaussian_Seidel(A, phi, u, stride, mu);
        return;
    }
    int N = stride;
    int n = u.size() / N;
    for(int nu = 0; nu < mu; ++nu) {
        for(int i = 0; i < n; ++i) {
            Mat_t Ai = A.block(i*N, i*N, N, N);
            Vec_t rhs_i = phi.segment(i*N, N) - A.block(i*N, 0, N, N*n) * u 
                + Ai * u.segment(i*N, N);
            u.segment(i*N, N) = Ai.colPivHouseholderQr().solve(rhs_i);
        }
    }
}

void block_GS(SpMat_t& A, Vec_t& phi, Vec_t& u, Vec_t& sol, int stride){
    // u: initial value;
    std::cout << std::left << std::setw(10) << "Iteration"
        << std::setw(20) << "Err_norm" << std::endl;

    int N = stride;
    int n = u.size() / N;

    int cnt = 0;
    while(true){
        cnt++;
        for(int i = 0; i < n; ++i) {
            Mat_t Ai = A.block(i*N, i*N, N, N);
            Vec_t rhs_i = phi.segment(i*N, N) - A.block(i*N, 0, N, N*n) * u 
                + Ai * u.segment(i*N, N);
            u.segment(i*N, N) = Ai.colPivHouseholderQr().solve(rhs_i);
        }
        double err = (sol - u).norm() / sol.norm();
        if(cnt % 20 == 0) {
            std::cout << std::left << std::setw(10) << cnt
                << std::setw(20) << err << std::endl;
        }
        if(err < 0.01){
            std::cout << "Block GS iteration converges after " << cnt << " iterations." << std::endl;
            break;
        }
        if(cnt >= 500) {
            std::cout << "block GS iteration doesn't converge after "
                      << cnt << " iterations." << std::endl; 
            break;
        }
    }
}

void Kaczmarz(SpMat_t& A, Vec_t& phi, Vec_t& u, int mu) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Eigen::VectorXd A_rowwise_norm = Mat_t(A).rowwise().squaredNorm();
    A_rowwise_norm = A_rowwise_norm / A_rowwise_norm.sum();

    int N = A.rows();
    for(int k = 0; k < mu; ++k) {
        // int i = k % N;
        int i = 0;
        double random_number = dis(gen), acc = 0.0;
        for( ; i < N; ++i) {
            acc += A_rowwise_norm(i);
            if(acc > random_number){
                break;
            }
        }
        Vec_t rowi_T = A.row(i).transpose();
        Vec_t tmp = u + (phi(i) - (u.conjugate()).dot(rowi_T)) / rowi_T.squaredNorm() * rowi_T.conjugate();
        u = tmp;
    }
}

void Kaczmarz(SpMat_t& A, Vec_t& phi, Vec_t& u, Vec_t& sol){
    // u: initial value;
    std::cout << std::left << std::setw(10) << "Iteration"
        << std::setw(20) << "Err_norm" << std::endl;
    int N = A.rows();
    int cnt = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Eigen::VectorXd A_rowwise_norm = Mat_t(A).rowwise().squaredNorm();
    A_rowwise_norm = A_rowwise_norm / A_rowwise_norm.sum();
    while(true){
        // i is selected with probability proportion to A.row(i).squaredNorm();
        int i = 0;
        double random_number = dis(gen), acc = 0.0;
        for( ; i < N; ++i) {
            acc += A_rowwise_norm(i);
            if(acc > random_number){
                break;
            }
        }

        Vec_t rowi_T = A.row(i).transpose();
        Vec_t tmp = u + (phi(i) - (u.conjugate()).dot(rowi_T)) / rowi_T.squaredNorm() * rowi_T.conjugate();
        u = tmp;
        
        cnt++;
        double err = (sol - u).norm() / sol.norm();
        if(cnt % 50 == 0) {
            std::cout << std::left << std::setw(10) << cnt
                << std::setw(20) << err << std::endl;
        }
        
        if(err < 0.01){
            std::cout << "Kaczmarz iteration converges after " << cnt << " iterations." << std::endl;
            break;
        }
        if(cnt >= 2000) {
            std::cout << "Kaczmarz iteration doesn't converge after "
                      << cnt << " iterations." << std::endl; 
            break;
        }
    }
}

std::pair<Vec_t, Scalar> power_GS(SpMat_t& A, int stride) {
    /* Compute the Eigen value of the GS operator manually */
    Mat_t dense_A = Mat_t(A);
    Mat_t L = Mat_t(dense_A.triangularView<Eigen::Lower>());
    Mat_t U = L - A;
    Mat_t GS_op = L.colPivHouseholderQr().solve(U);
    Vec_t eivals = GS_op.eigenvalues();

    Scalar domainant_eival = eivals(0);
    for(int i = 1; i < eivals.size(); ++i) {
        if(std::abs(eivals(i)) > std::abs(domainant_eival)) {
            domainant_eival = eivals(i);
        }
    }
    // std::cout << eivals << std::endl;
    std::cout << "Domainant eigenvalue: " << domainant_eival << std::endl;
    std::cout << "Absolute value: " << std::abs(domainant_eival) << std::endl;
    /**********************************************/

    double tol = 0.0001;
    int N = A.rows();
    Vec_t u = Vec_t::Random(N);

    u.normalize();    
    Scalar lambda;
    int cnt = 0;

    std::cout << std::left << std::setw(10) << "Iteration"
        << std::setw(20) << "residual_norm" << std::endl;
    while(1){
        cnt++;
        Vec_t old_u = u;
        for(int t = 0; t < stride; ++t) {
            for(int k = 0; k < N / stride; ++k) {
                int j = k * stride + t;
                Scalar tmp = (A.row(j) * u)(0,0);
                Scalar Ajj = A.coeffRef(j,j);
                u(j) = (u(j) * Ajj - tmp) / Ajj;
            }
        }
        // now u should be GS_op * old_u
        lambda = old_u.dot(u); // Rayleigh quotient
        // compute the residual and check vs tolerance
        auto r = u - lambda * old_u;
        double r_norm = r.norm();
        if(cnt % 20 == 0){
            std::cout << std::left << std::setw(10) << cnt
                << std::setw(20) << r_norm << std::endl;
        }

        u.normalize();
        
        if(r_norm < tol) {
            std::cout << "Power iteration for Gauss-Seidel converges after " << cnt 
                << " iterations." << std::endl;
            break;
        }
        if(cnt >= 500) {
            std::cout << "Power iteration for Gauss-Seidel doesn't converge after " << cnt 
                << " iterations." << std::endl; 
            break;
        }
    }
    std::cout << "Number of iterations: " << cnt << std::endl;
    std::cout << "Domainant eigenvalue of Gauss-Seidel by power iteration: " << std::abs(lambda) << std::endl;
    return std::make_pair(u, lambda);
}

std::pair<Vec_t, Scalar> power_GS(Mat_t& A, int stride) {
    /* Compute the Eigen value of the GS operator manually */
    Mat_t A_L = A.triangularView<Eigen::Lower>();
    Mat_t A_U = A_L - A;
    Mat_t GS_op = A_L.colPivHouseholderQr().solve(A_U);
    Vec_t eivals = GS_op.eigenvalues();

    Scalar domainant_eival = eivals(0);
    for(int i = 1; i < eivals.size(); ++i) {
        if(std::abs(eivals(i)) > std::abs(domainant_eival)) {
            domainant_eival = eivals(i);
        }
    }
    // std::cout << eivals << std::endl;
    std::cout << "Domainant eigenvalue: " << domainant_eival << std::endl;
    std::cout << "Absolute value: " << std::abs(domainant_eival) << std::endl;
    /**********************************************/

    double tol = 0.0001;
    int N = A.rows();
    Vec_t u = Vec_t::Random(N);

    u.normalize();    
    Scalar lambda;
    int cnt = 0;

    std::cout << std::left << std::setw(10) << "Iteration"
        << std::setw(20) << "residual_norm" << std::endl;
    while(1){
        cnt++;
        Vec_t old_u = u;
        for(int t = 0; t < stride; ++t) {
            for(int k = 0; k < N / stride; ++k) {
                int j = k * stride + t;
                Scalar tmp = (A.row(j) * u)(0,0);
                Scalar Ajj = A(j,j);
                u(j) = (u(j) * Ajj - tmp) / Ajj;
            }
        }
        // now u should be GS_op * old_u
        lambda = old_u.dot(u); // Rayleigh quotient
        // compute the residual and check vs tolerance
        auto r = u - lambda * old_u;
        double r_norm = r.norm();
        if(cnt % 20 == 0){
            std::cout << std::left << std::setw(10) << cnt
                << std::setw(20) << r_norm << std::endl;
        }

        u.normalize();
        
        if(r_norm < tol) {
            std::cout << "Power iteration for Gauss-Seidel converges after " << cnt 
                << " iterations." << std::endl;
            break;
        }
        if(cnt >= 500) {
            std::cout << "Power iteration for Gauss-Seidel doesn't converge after " << cnt 
                << " iterations." << std::endl; 
            break;
        }
    }
    std::cout << "Number of iterations: " << cnt << std::endl;
    std::cout << "Domainant eigenvalue of Gauss-Seidel by power iteration: " << std::abs(lambda) << std::endl;
    return std::make_pair(u, lambda);
}

std::pair<Vec_t, Scalar> power_block_GS(SpMat_t& A, int stride) {
    
    double tol = 0.001;
    Vec_t u = Vec_t::Random(A.rows());

    u.normalize();    
    Scalar lambda;
    int cnt = 0;

    int N = stride;
    int n = u.size() / N;

    std::cout << std::left << std::setw(10) << "Iteration"
        << std::setw(20) << "residual_norm" << std::endl;
    while(1){
        cnt++;
        Vec_t old_u = u;

        for(int i = 0; i < n; ++i) {
            Mat_t Ai = A.block(i*N, i*N, N, N);
            Vec_t rhs_i = - A.block(i*N, 0, N, N*n) * u + Ai * u.segment(i*N, N);
            u.segment(i*N, N) = Ai.colPivHouseholderQr().solve(rhs_i);
        }
        
        lambda = old_u.dot(u); // Rayleigh quotient
        auto r = u - lambda * old_u;
        double r_norm = r.norm();
        if(cnt % 20 == 0){
            std::cout << std::left << std::setw(10) << cnt
                << std::setw(20) << r_norm << std::endl;
        }

        u.normalize();
        
        if(r_norm < tol) {
            std::cout << "Power iteration for block GS converges after " << cnt 
                << " iterations." << std::endl;
            break;
        }
        if(cnt >= 500) {
            std::cout << "Power iteration for block GS doesn't converge after " << cnt 
                << " iterations." << std::endl; 
            break;
        }
    }
    std::cout << "Number of iterations: " << cnt << std::endl;
    std::cout << "Domainant eigenvalue of block GS by power iteration: " << lambda << std::endl;
    return std::make_pair(u, lambda);
}

std::pair<Vec_t, Scalar> power_kaczmarz(SpMat_t& A) {
    int N = A.rows();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Eigen::VectorXd A_rowwise_norm = Mat_t(A).rowwise().squaredNorm();
    A_rowwise_norm = A_rowwise_norm / A_rowwise_norm.sum();

    double tol = 0.001;
    Vec_t u = Vec_t::Random(N);

    u.normalize();    
    Scalar lambda;
    int cnt = 0;

    std::cout << std::left << std::setw(10) << "Iteration"
        << std::setw(20) << "residual_norm" << std::endl;
    while(1){
        cnt++;
        Vec_t old_u = u;

        int i = 0;
        double random_number = dis(gen), acc = 0.0;
        for( ; i < N; ++i) {
            acc += A_rowwise_norm(i);
            if(acc > random_number){
                break;
            }
        }
        Vec_t rowi_T = A.row(i).transpose();
        Vec_t tmp = u + (-(u.conjugate()).dot(rowi_T)) / rowi_T.squaredNorm() * rowi_T.conjugate();
        u = tmp;
        
        lambda = old_u.dot(u); // Rayleigh quotient
        auto r = u - lambda * old_u;
        double r_norm = r.norm();
        if(cnt % 20 == 0){
            std::cout << std::left << std::setw(10) << cnt
                << std::setw(20) << r_norm << std::endl;
        }

        u.normalize();
        
        if(r_norm < tol) {
            std::cout << "Power iteration for Kaczmarz iteration converges after " << cnt 
                << " iterations." << std::endl;
            break;
        }
        if(cnt >= 500) {
            std::cout << "Power iteration for Kaczmarz iteration doesn't converge after " << cnt 
                << " iterations." << std::endl; 
            break;
        }
    }
    std::cout << "Number of iterations: " << cnt << std::endl;
    std::cout << "Domainant eigenvalue of Kaczmarz iteration by power iteration: " << lambda << std::endl;
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
 * nu1, nu1: pre and post relaxation times
 */
void v_cycle(Vec_t& u, Vec_t& f, std::vector<SpMat_t>& Op, std::vector<SpMat_t>& I, 
    std::vector<int>& stride, int nu1, int nu2, bool solve_on_coarest) {

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
    for(int i = L; i > 0; --i) {
        Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], stride[i], nu1);
        rhs_vec[i-1] = I[i-1].transpose() * (rhs_vec[i] - Op[i] * initial[i]);
    }

    if(solve_on_coarest) {
        Eigen::SparseLU<SpMat_t> solver;
        solver.compute(Op[0]);
        initial[0] = solver.solve(rhs_vec[0]);
    } else {
        Gaussian_Seidel(Op[0], rhs_vec[0], initial[0], stride[0], nu1 + nu1);
    }

    for(int i = 1; i <= L; ++i) {
        initial[i] += I[i-1] * initial[i-1];
        Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], stride[i], nu1);
    }
    u = initial[L];
}

/*
 * return the convergence factor of the multigrid method
 */
void mg_factor(HE_FEM& he_fem, int L, int nr_coarsemesh, double k, 
    std::vector<int>& stride, FHandle_t u, bool solve_coarest) {
    auto eq_pair = he_fem.build_equation(L);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(nr_coarsemesh + 1), prolongation_op(nr_coarsemesh);
    std::vector<double> ms(nr_coarsemesh + 1);
    auto mesh_width = he_fem.mesh_width();
    Op[nr_coarsemesh] = A;
    ms[nr_coarsemesh] = mesh_width[L];
    for(int i = nr_coarsemesh - 1; i >= 0; --i) {
        int idx = L + i - nr_coarsemesh;
        prolongation_op[i] = he_fem.prolongation(idx);
        auto tmp = he_fem.build_equation(idx);
        Op[i] = tmp.first.makeSparse();
        // Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
        ms[i] = mesh_width[idx];
    }

    /**************************************************************************/
    auto zero_fun = [](const coordinate_t& x) -> Scalar { return 0.0; };

    int N = A.rows();

    Vec_t v = Vec_t::Random(N); // initial value
    Vec_t uh = he_fem.solve(L); // finite element solution

    int nu1 = 1, nu2 = 1;

    std::vector<double> L2_vk;
    std::vector<double> L2_ek;  // error norm

    // std::cout << std::scientific << std::setprecision(1);
    std::cout << std::setw(11) << "||v-uh||_2" << std::setw(11) << "||v-u||_2" << std::endl;
    for(int k = 0; k < 10; ++k) {
        std::cout << std::setw(11) << he_fem.L2_Err(L, v - uh, zero_fun) << " ";
        std::cout << std::setw(11) << he_fem.L2_Err(L, v, u) << std::endl;
        L2_vk.push_back(he_fem.L2_Err(L, v, zero_fun));
        L2_ek.push_back(he_fem.L2_Err(L, v - uh, zero_fun));
        v_cycle(v, eq_pair.second, Op, prolongation_op, stride, nu1, nu2, solve_coarest);
    }

    std::cout << "||u-uh||_2 = " << he_fem.L2_Err(L, uh, u) << std::endl;
    std::cout << "k " 
        << std::setw(20) << "||v_{k+1}||/||v_k||" 
        << std::setw(20) << "||e_{k+1}||/||e_k||" << std::endl;
    std::cout << std::left;
    for(int k = 0; k + 1 < L2_vk.size(); ++k) {
        std::cout << k << " " << std::setw(20) << L2_vk[k+1] / L2_vk[k] 
                       << " " << std::setw(20) << L2_ek[k+1] / L2_ek[k] << std::endl;
    }
} 

std::pair<Vec_t, Scalar> 
power_multigird(HE_FEM& he_fem, int start_layer, int num_coarserlayer, 
    std::vector<int>& stride, int nu1, int nu2, bool verbose) {

    LF_ASSERT_MSG((num_coarserlayer <= start_layer), 
        "please use a smaller number of wave layers");

    auto eq_pair = he_fem.build_equation(start_layer);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(num_coarserlayer + 1), prolongation_op(num_coarserlayer);
    Op[num_coarserlayer] = A;
    for(int i = num_coarserlayer - 1; i >= 0; --i) {
        int idx = start_layer + i - num_coarserlayer;
        prolongation_op[i] = he_fem.prolongation(idx);
        Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
    }

    /***************************************/
    int N = A.rows();
    Vec_t u = Vec_t::Random(N);
    u.normalize();
    Vec_t old_u;
    Vec_t zero_vec = Vec_t::Zero(N);
    Scalar lambda;
    int cnt = 0;
    
    if(verbose) {
        std::cout << std::left << std::setw(10) << "Iteration" 
            << std::setw(20) << "residual_norm" << std::endl;
    }
    
    while(true) {
        cnt++;
        old_u = u;
        v_cycle(u, zero_vec, Op, prolongation_op, stride, nu1, nu2);
        
        lambda = old_u.dot(u);  // domainant eigenvalue
        auto r = u - lambda * old_u;
        double r_norm = r.norm();
        u.normalize();
    
        if(verbose && cnt % 5 == 0) {
            std::cout << std::left << std::setw(10) << cnt 
                << std::setw(20) << r_norm
                << std::setw(20) << (u - old_u).norm()
                << std::endl;
        }
        
        if(r_norm < 0.01) {
            if(verbose) {
                std::cout << "Power iteration converges after " << cnt << " iterations" << std::endl;
            }
            break;
        }
        if(cnt > 30) {
            if(verbose) {
                std::cout << "Power iteration for multigrid doesn't converge." << std::endl;
            }
            break;
        }
    }
    if(verbose) {
        std::cout << "Number of iterations: " << cnt << std::endl;
        std::cout << "Domainant eigenvalue by power iteration: " << lambda << std::endl;
    }
    return std::make_pair(u, lambda);
}