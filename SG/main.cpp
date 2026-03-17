#include <sgpp/base.hpp>
#include <sgpp/pde.hpp>
#include <sgpp/solver.hpp>
#include <cmath>
#include <vector>
#include <iostream>

using namespace sgpp::base;
using namespace sgpp::pde;
using namespace sgpp::solver;

// Define PDE parameters
const size_t d = 7;             // Dimension
const size_t level = 4;         // Sparse grid level
const double delta_diff = 0.1;  // Diffusion coefficient
const double alpha = 1.0;
const double beta = 0.5;

// Vectors for the analytical Gaussian packet
std::vector<double> a(d, 1.0);
std::vector<double> b_vec(d, 0.5);
std::vector<double> c(d, 0.1);

// Reaction coefficient w = -2 * delta * alpha * sum(a_i^2)
double compute_w() {
    double sum_a2 = 0.0;
    for(size_t i=0; i<d; ++i) sum_a2 += a[i]*a[i];
    return -2.0 * delta_diff * alpha * sum_a2;
}

// Analytical exact solution for projecting the initial condition
double exact_solution(const DataVector& x, double t) {
    double sum_term = 0.0;
    for(size_t i=0; i<d; ++i) {
        double term = a[i]*x[i] - b_vec[i] + c[i]*t;
        sum_term += term * term;
    }
    return std::exp(-alpha * sum_term) * std::exp(-beta * t);
}

// -----------------------------------------------------------------
// Matrix-Free System Operator: A = (M + theta * dt * L)
// -----------------------------------------------------------------
class SystemOperator : public OperationMatrix {
private:
    std::unique_ptr<OperationMatrix> opMass;
    std::unique_ptr<OperationMatrix> opLaplace;
    double dt, theta, w;

public:
    SystemOperator(Grid& grid, double dt_in, double theta_in) 
        : dt(dt_in), theta(theta_in), w(compute_w()) {
        
        // SG++ factory methods to create standard 1D-tensor-product operators
        opMass.reset(sgpp::op_factory::createOperationIdentity(grid));
        opLaplace.reset(sgpp::op_factory::createOperationLaplace(grid));
    }

    // Implements y = A * x
    void mult(const DataVector& x, DataVector& y) override {
        DataVector temp_M(x.getSize());
        DataVector temp_L(x.getSize());
        
        // 1. Compute Mass contribution: temp_M = M * x
        opMass->mult(x, temp_M);
        
        // 2. Compute Laplace contribution: temp_L = S * x
        opLaplace->mult(x, temp_L);
        
        // Note: For a complete implementation of convection (v * nabla u)
        // and reaction (w * u), one would typically use 
        // sgpp::op_factory::createOperationBilinearForm or custom quadratures here.
        // For brevity, we approximate L * x approx delta_diff * S * x + w * M * x
        
        // temp_L = delta * S * x + w * M * x (combining diffusion and reaction)
        for(size_t i=0; i<temp_L.getSize(); ++i) {
            temp_L[i] = delta_diff * temp_L[i] + w * temp_M[i];
        }

        // 3. Combine to form the implicit system matrix evaluation
        // y = M*x + theta * dt * L*x
        for(size_t i=0; i<y.getSize(); ++i) {
            y[i] = temp_M[i] + theta * dt * temp_L[i];
        }
    }
};

int main() {
    // 1. Initialize 7D Grid without boundaries (functions vanish at infinity/boundaries)
    std::unique_ptr<Grid> grid(Grid::createLinearGrid(d));
    GridGenerator& gridGen = grid->getGenerator();
    gridGen.regular(level);
    
    size_t num_dof = grid->getSize();
    std::cout << "Sparse Grid Dimension d = " << d << ", Level = " << level << std::endl;
    std::cout << "Degrees of freedom: " << num_dof << std::endl;

    // 2. Initialize Hierarchical Coefficients DataVector
    DataVector gamma(num_dof);
    gamma.setAll(0.0);

    // L2-Projection of Initial Condition t=0 onto the hierarchical basis
    // SG++ provides OperationL2Projection to compute the initial coefficients
    auto opProj = std::unique_ptr<OperationMatrix>(
        sgpp::op_factory::createOperationL2Projection(*grid));
    
    // Create a function wrapper for the initial condition
    auto f_init = [](int d, const double* x, void* clientData) -> double {
        DataVector pt(d);
        for(int i=0; i<d; ++i) pt[i] = x[i];
        return exact_solution(pt, 0.0);
    };
    
    // Project exact solution at t=0 to get gamma_0
    // (Implementation relies on SG++'s quadrature routines, omitted here for brevity)
    // opProj->mult(..., gamma); 

    // 3. Setup Time Stepping (Implicit Euler: theta = 1.0)
    double t = 0.0;
    double t_end = 1.0;
    double dt = 0.01;
    double theta = 1.0; 

    // Instantiate our custom matrix-free system operator
    SystemOperator sysOp(*grid, dt, theta);

    // Setup the BiCGStab Solver for asymmetric operators
    size_t max_iters = 100;
    double tolerance = 1e-6;
    BiCGStab solver(max_iters, tolerance);

    // Pre-allocate vectors for the RHS and solver
    DataVector rhs(num_dof);
    DataVector load_vector(num_dof); // Evaluated f(x,t)
    
    // 4. Time Integration Loop
    while (t < t_end) {
        // In a full implementation, project the source term f(x, t_{m+1}) 
        // onto 'load_vector' using quadrature here.
        load_vector.setAll(0.0); // Simplified placeholder

        // Construct the Right Hand Side: b = M * gamma_m + dt * f_m+1
        // For Implicit Euler (theta=1), the previous step contribution is just M * gamma_m
        auto opMass = std::unique_ptr<OperationMatrix>(
            sgpp::op_factory::createOperationIdentity(*grid));
        opMass->mult(gamma, rhs);

        for(size_t i=0; i<num_dof; ++i) {
            rhs[i] += dt * load_vector[i];
        }

        // Solve the linear system A * gamma_{m+1} = rhs
        // sysOp evaluates (M + dt * L) matrix-free
        solver.solve(sysOp, rhs, gamma, true, false);

        t += dt;
        std::cout << "Time t = " << t << ", Solver Iters = " << solver.getNumberIterations() << std::endl;
    }

    std::cout << "Time integration complete." << std::endl;
    return 0;
}