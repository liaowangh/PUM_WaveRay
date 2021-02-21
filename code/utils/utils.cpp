#include <fstream>
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
    // double res = 0.0;
    // for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
    //     res += std::abs(LocalIntegral(*cell, degree, f_square));
    // }
    // return std::sqrt(res);
}

void print_save_error(std::vector<std::vector<double>>& data, 
    std::vector<std::string>& data_label, const std::string& sol_name, 
    const std::string& output_folder) {
    
    std::cout << sol_name << std::endl;
    std::cout << std::left;
    for(int i = 0; i < data_label.size(); ++i){
        std::cout << std::setw(10) << data_label[i];
    }
    std::cout << std::endl;
    std::cout << std::left << std::scientific << std::setprecision(1);
    for(int l = 0; l < data[0].size(); ++l) {
        for(int i = 0; i < data.size(); ++i) {
            std::cout << std::setw(10) << data[i][l];
        }
        std::cout << std::endl;
    }

    // write the result to the file
    std::string output_file = output_folder + sol_name + ".txt";
    std::ofstream out(output_file);

    out << data_label[0];
    for(int i = 1; i < data_label.size(); ++i) {
        out << " " << data_label[i];
    }
    out << std::endl;
    for(int l = 0; l < data[0].size(); ++l) {
        out << data[0][l];
        for(int i = 1; i < data.size(); ++i) {
            out << " " << data[i][l];
        }
        out << std::endl;
    } 
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
    std::cout << std::left << std::scientific << std::setprecision(1);
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
    const std::string& output_folder, size_type L, const FHandle_t& u, 
    const FunGradient_t& grad_u) {
    // std::vector<int> ndofs;
    std::vector<double> mesh_width = he_fem.mesh_width();
    std::vector<double> L2err, H1serr, H1err;
    
    for(size_type level = 0; level <= L; ++level) {
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

void test_multigrid(HE_FEM& he_fem, int num_coarserlayer, const std::string& sol_name, 
    const std::string& output_folder, size_type L, const FHandle_t& u,
    const FunGradient_t& grad_u) {

    // std::vector<int> ndofs;
    std::vector<double> mesh_width = he_fem.mesh_width();
    std::vector<double> L2err, H1serr, H1err;
    
    for(size_type level = num_coarserlayer; level <= L; ++level) {
        auto fe_sol = he_fem.solve_multigrid(level, num_coarserlayer, 10, 10);
        
        double l2_err = he_fem.L2_Err(level, fe_sol, u);
        double h1_serr = he_fem.H1_semiErr(level, fe_sol, grad_u);
        double h1_err = std::sqrt(l2_err*l2_err + h1_serr*h1_serr);
        
        // ndofs.push_back(fe_sol.size());
        L2err.push_back(l2_err);
        H1serr.push_back(h1_serr);
        H1err.push_back(h1_err);
    }
    int num_grids = 1 + num_coarserlayer;
    std::vector<std::vector<double>> err_data{mesh_width, L2err, H1err, H1serr};
    std::vector<std::string> err_str{"h", "L2_err", "H1_err", "H1_serr"};
    std::string sol_name_mg = std::to_string(num_grids) + "grids_" + sol_name;
    print_save_error(err_data, err_str, sol_name_mg, output_folder);
}

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
        if(cnt % 20 == 0) {
            std::cout << std::left << std::setw(10) << cnt
                << std::setw(20) << (sol - u).norm() / sol.norm() << std::endl;
        }
        
        if((sol - u).norm() < 0.01){
            break;
        }
        if(cnt > 500) {
            std::cout << "Gaussian Seidel iteration doesn't converge after "
                      << cnt << " iterations." << std::endl; 
            break;
        }
    }
}

std::pair<Vec_t, Scalar> power_GS(SpMat_t& A, int stride) {
    /* Compute the Eigen value of the GS operator */
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
    std::cout << eivals << std::endl;
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

void Kaczmarz(SpMat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu) {
    int N = A.rows();
    for(int k = 0; k < mu; ++k) {
        int i = k % N;
        Vec_t rowi_T = A.row(i).transpose();
        Vec_t tmp = u + (phi(i) - u.dot(rowi_T)) / rowi_T.squaredNorm() * rowi_T.conjugate();
        u = tmp;
    }
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
void v_cycle(Vec_t& u, Vec_t& f, std::vector<SpMat_t>& Op, std::vector<SpMat_t>& I, 
    std::vector<int>& stride, size_type mu1, size_type mu2) {

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
        Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], stride[i], mu1);
        rhs_vec[i-1] = I[i-1].transpose() * (rhs_vec[i] - Op[i] * initial[i]);
    }

    Eigen::SparseLU<SpMat_t> solver;
    solver.compute(Op[0]);
    initial[0] = solver.solve(rhs_vec[0]);
    // Gaussian_Seidel(Op[0], rhs_vec[0], initial[0], stride[0], mu1 + mu2);

    for(int i = 1; i <= L; ++i) {
        initial[i] += I[i-1] * initial[i-1];
        Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], stride[i], mu2);
    }
    u = initial[L];
}


/*
 * Return the multigrid operator
 * 
 * For a 2-grid correction scheme
 *  mg_op = R^{mu2}*(Id - I_H^h * A_H^{-1} * I_h^H * A_h) * R^{mu1}
 * 
 * l : indicator level, useful in a recursive method
 * Op: mesh operator
 * I: prolongation operator
 * R: relaxation operator
 * mu1, mu2: pre and post relaxation 
 */
// Mat_t multigrid_op(int l, std::vector<SpMat_t>& Op, std::vector<SpMat_t>& I, size_type mu1, size_type mu2) {
//     int N = Op[l].rows();
//     Mat_t Id = Mat_t::Identity(N, N);
//     if(l == 0) {
//         return Op[0].colPivHouseholderQr().solve(Id);
//     } else {
//         Mat_t A = Mat_t(Op[l]);
//         Mat_t L = Mat_t(A.triangularView<Eigen::Lower>());
//         Mat_t U = L - A;
//         Mat_t GS_op = L.colPivHouseholderQr().solve(U);
// 
//         Mat_t R_mu1 = Mat_t::Identity(N, N);
//         Mat_t R_mu2 = Mat_t::Identity(N, N);
//         for(int i = 0; i < mu1; ++i) {
//             auto tmp = R_mu1 * GS_op;
//             R_mu1 = tmp;
//         }
//         for(int i = 0; i < mu2; ++i) {
//             auto tmp = R_mu2 * GS_op;
//             R_mu2 = tmp;
//         }
//         Mat_t mg_op = Id - I[l] * multigrid_op(l-1, Op, I, mu1, mu2) *  
//         auto tmp = R_mu2 * mg_op * R_mu1;
// 
//         return Id - I[l] * multigrid_op(l-1, Op, I, mu1, mu2) * 
//     }
// }