#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <Eigen/Dense>
#include <cassert>
#include <omp.h>


using namespace std;
using namespace Eigen;

// Function declarations
void writeSolutionToVTK(const string& filename, 
    const vector<double>& x_coords, 
    const vector<double>& y_coords, 
    const vector<vector<int>>& connectivity, 
    const vector<double>& u, 
    const vector<double>& v,
    const vector<double>& p); 

double summed_multiplication(const vector<double>& N, const vector<double>& xp);
vector<vector<double>> maping(const vector<double>& xcoord, const vector<double>& ycoord, 
                           double zita, double eta, double &determinent);

// Boundary condition functions
bool onLeftBoundary(int i, int Nx, int Ny) {
    return (i % Nx) == 0;
}
bool onRightBoundary(int i, int Nx, int Ny) {
    return (i % Nx) == Nx - 1;
}
bool onBottomBoundary(int i, int Nx, int Ny) {
    return i < Nx;
}
bool onTopBoundary(int i, int Nx, int Ny) {
    return i >= (Ny - 1) * Nx;
}

// Natural integration constants
const vector<double> w = {8.0/9.0, 5.0/9.0, 5.0/9.0};
const vector<double> point_natural{0, sqrt(3.0/5.0), -sqrt(3.0/5.0)};

// Simulation parameters
const int Ndt = 3002;
constexpr double dt = 0.001 ;
constexpr double Re = 100.0;

int main() {
    double start_time = omp_get_wtime();
    // Geometry parameters
    const int Ex = 20, Ey = 20;     // Number of elements
    const double Lx = 1.0, Ly = 1.0; // Domain size
    const double x_min = 0.0, y_min = 0.0;
    const double epsilon = 0.01;

    // Derived parameters
    const int nnx = Ex + 1;
    const int nny = Ey + 1;
    const double dx = Lx / Ex;
    const double dy = Ly / Ey;
    const int NT = nnx * nny;
    const int ET = Ex * Ey;
    const double le = dx;

    // Coordinate arrays
    vector<double> x_coords(NT);
    vector<double> y_coords(NT);

    // Node generation
    for (int j = 0; j < nny; ++j) {
        for (int i = 0; i < nnx; ++i) {
            int idx = i + j * nnx;
            x_coords[idx] = x_min + i * dx;
            y_coords[idx] = y_min + j * dy;
        }
    }

    // Connectivity matrix
    vector<vector<int>> connectivity(ET, vector<int>(4));
    for (int j = 0; j < Ey; ++j) {
        for (int i = 0; i < Ex; ++i) {
            int en = i + j * Ex;
            int n1 = i + j * nnx;
            connectivity[en] = {n1, n1 + 1, n1 + 1 + nnx, n1 + nnx};
        }
    }
    MatrixXd M_lumped = MatrixXd::Zero(NT, NT);
    MatrixXd K_0 = MatrixXd::Zero(NT, NT);
    MatrixXd K_1 = MatrixXd::Zero(NT, NT);
    MatrixXd K_2 = MatrixXd::Zero(NT, NT);
    MatrixXd M = MatrixXd::Zero(NT, NT);
    double determinent = 0.0;

    // Create global matrices
    for (int element = 0; element < ET; element++) {
        auto conn = connectivity[element];
        vector<double> pointsx = {x_coords[conn[0]], x_coords[conn[1]], x_coords[conn[2]], x_coords[conn[3]]};
        vector<double> pointsy = {y_coords[conn[0]], y_coords[conn[1]], y_coords[conn[2]], y_coords[conn[3]]};
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double p1 = point_natural[i];
                double p2 = point_natural[j];
                vector<vector<double>> result = maping(pointsx, pointsy, p1, p2, determinent);
                vector<double> Ny = result[0];
                vector<double> Nx = result[1];
                vector<double> N = result[2];

                for (int a = 0; a < 4; a++) {
                    for (int b = 0; b < 4; b++) {
                        K_0(conn[a], conn[b]) += (Nx[a]*Nx[b] + Ny[a]*Ny[b]) * determinent * w[i] * w[j];
                        K_1(conn[a], conn[b]) += N[a]*Nx[b] * determinent * w[i] * w[j];
                        K_2(conn[a], conn[b]) += N[a]*Ny[b] * determinent * w[i] * w[j];
                        M(conn[a], conn[b]) += N[b]*N[a] * determinent * w[i] * w[j]/dt;
                    }
                }
            }
        }
       
    }
    

    for (int i = 0; i < M.rows(); ++i) {
        double rowSum = M.row(i).sum();  // Sum of the i-th row
        M_lumped(i, i) = rowSum;         // Place it in the diagonal
    }
    
    cout << "getting inverse matrix"<<endl;
   

    MatrixXd K_inv = K_0.fullPivLu().inverse();
    // Initialize solution vectors
    VectorXd u_old = VectorXd::Zero(NT);
    VectorXd v_old = VectorXd::Zero(NT);
    VectorXd p_old = VectorXd::Zero(NT);

    // Apply initial boundary conditions
    
    for (int i = 0; i < NT; ++i) {
        if (onLeftBoundary(i, nnx, nny) || onRightBoundary(i, nnx, nny) || onBottomBoundary(i, nnx, nny)) {
            u_old(i) = 0.0;
            v_old(i) = 0.0;
        }
        if (onTopBoundary(i, nnx, nny)) {
            u_old(i) = 1.0;  // Lid-driven cavity
            v_old(i) = 0.0;
        }
        if (i == nnx*nny-1 || i == nnx*(nny-1)) {
            u_old(i) = 0.5;  
        }
       
    }
    
    
    // Time stepping loop
    for (int t = 0; t < Ndt; ++t) {
        cout << "Starting time step: " << t << endl;
     
        VectorXd RHS_cont = VectorXd::Zero(NT);
        VectorXd RHS_xmom = VectorXd::Zero(NT);
        VectorXd RHS_ymom = VectorXd::Zero(NT);
        MatrixXd K_3 = MatrixXd::Zero(NT, NT);
        
        for (int element = 0; element < ET; element++) {
            auto conn = connectivity[element];
            vector<double> pointsx = {x_coords[conn[0]], x_coords[conn[1]], x_coords[conn[2]], x_coords[conn[3]]};
            vector<double> pointsy = {y_coords[conn[0]], y_coords[conn[1]], y_coords[conn[2]], y_coords[conn[3]]};
            
           
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 3; ++i) {
                    double p1 = point_natural[i];
                    double p2 = point_natural[j];
                    vector<vector<double>> result = maping(pointsx, pointsy, p1, p2, determinent);
                    vector<double> Ny = result[0];
                    vector<double> Nx = result[1];
                    vector<double> N = result[2];
                    double ue = 0, ve = 0; 
                    for (int z = 0; z < 4; ++z) {
                        ue +=   u_old(conn[z]) * N[z] ;
                        ve +=   v_old(conn[z]) * N[z] ;
                    }  
                   // cout << " velocity  : " << ue <<"   /   " << ve << endl; 
                    for (int a = 0; a < 4; ++a) {
                        for (int b = 0; b < 4; ++b) {
                                double Ve = sqrt(pow(u_old(conn[b]),2) + pow(v_old(conn[b]),2));
                                double Tau = 1.0 / sqrt(pow(2.0/dt, 2) + pow(2.0*abs(Ve)/le, 2) + 
                                                pow(4.0/(Re*le*le), 2));
                                K_3(conn[a], conn[b]) += (ue*Nx[b] + ve *Ny[b]) * (N[a] + Tau * (ue* Nx[a] + ve* Ny[a]))* determinent * w[i] * w[j];
                        }
                    }
                }
            } 
        }   
            for ( int x = 0; x < NT; x++)
            {
                for (int y = 0; y < NT; y++)
                {
                    RHS_cont[x] -= (1.0/epsilon) * (K_1(x, y) * u_old(y) + K_2(x, y) * v_old(y));
                    RHS_xmom[x] += -1.0*K_1(x, y) * p_old(y) + (M_lumped(x, y)- K_3(x, y) - (1.0/Re)*K_0(x, y)) * u_old(y);        
                    RHS_ymom[x] += -1.0*K_2(x, y) * p_old(y) + (M_lumped(x, y) - K_3(x, y) - (1.0/Re)*K_0(x, y)) * v_old(y);
        
                } 
           // cout << RHS_cont(x) << "  /  " << RHS_xmom(x)<< "  /  " << RHS_ymom(x)<< endl ;
            }

        
        
        // Pressure solve
        cout <<"pressure solve" <<endl;

        VectorXd p_new = K_inv * RHS_cont;

        cout << "pressure solved "<< endl;
        cout<< "Reapply boundary conditions" << endl;
        for (int i = 0; i < NT; ++i) {
             p_old(i) = p_new(i);
        // Velocity update
            if (!onLeftBoundary(i, nnx, nny) && !onRightBoundary(i, nnx, nny) && 
                !onBottomBoundary(i, nnx, nny) && !onTopBoundary(i, nnx, nny)) {
                
                u_old(i) = RHS_xmom(i) / M_lumped(i, i);
                v_old(i) = RHS_ymom(i) / M_lumped(i, i);
            }

            if (onLeftBoundary(i, nnx, nny) || onRightBoundary(i, nnx, nny) || onBottomBoundary(i, nnx, nny)) {
                u_old(i) = 0.0;
                v_old(i) = 0.0;
            }
            if (onTopBoundary(i, nnx, nny)) {
                u_old(i) = 1.0;  // Lid-driven cavity
                v_old(i) = 0.0;
            }
            if (i == nnx*nny-1 || i == nnx*(nny-1)) {
            u_old(i) = 0.5;  
        }
            
           
            
        }
      

        // Output results
        if (t % 100 == 0) {
           string filename = "fineTime05_10x10_" + to_string(t) + ".vtk";

            vector<double> u_vec(u_old.data(), u_old.data() + u_old.size());
            vector<double> v_vec(v_old.data(), v_old.data() + v_old.size());
            vector<double> p_vec(p_old.data(), p_old.data() + p_old.size());

            writeSolutionToVTK(filename, x_coords, y_coords, connectivity, u_vec, v_vec, p_vec);

            cout << "VTK file written at time step " << t << endl;
            
            // Force calculation 
            double fx = 0.0, fy = 0.0; // Make sure to reset each time step

            for (int element = 0; element < ET; element++) {
                if (onBottomBoundary(element, nnx, nny)) {
                    auto conn = connectivity[element];
                    vector<double> pointsx = {x_coords[conn[0]], x_coords[conn[1]], x_coords[conn[2]], x_coords[conn[3]]};
                    vector<double> pointsy = {y_coords[conn[0]], y_coords[conn[1]], y_coords[conn[2]], y_coords[conn[3]]};

                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            double p1 = point_natural[i];
                            double p2 = point_natural[j];

                            auto result = maping(pointsx, pointsy, p1, p2, determinent);
                            auto Ny = result[0];
                            auto Nx = result[1];
                            auto N  = result[2];

                            double uxe=0, uye=0, vxe=0, vye=0;
                            for (int z = 0; z < 4; ++z) {
                                uxe += u_old(conn[z]) * Nx[z];
                                uye += u_old(conn[z]) * Ny[z];
                                vxe += v_old(conn[z]) * Nx[z];
                                vye += v_old(conn[z]) * Ny[z];
                            }

                            double p = p_old[element];
                            Matrix2d sigma;
                            sigma(0,0) = -p + (1.0/Re)*(2*uxe);
                            sigma(0,1) = (1.0/Re)*(uye + vxe);
                            sigma(1,0) = sigma(0,1);
                            sigma(1,1) = -p + (1.0/Re)*(2*vye);

                            Vector2d n(0.0, -1.0); // bottom boundary normal
                            Vector2d traction = sigma * n;

                            fx += traction(0) * determinent * w[i] * w[j];
                            fy += traction(1) * determinent * w[i] * w[j];
                        }
                    }
                }
            }

            // Write forces to text file
            string force_filename = "fineTime05_10x10_" + to_string(t) + ".txt";
            ofstream force_file(force_filename);
            cout << "Time step: " << t << "\n";
            cout << "Fx = " << fx << "\n";
            cout << "Fy = " << fy << "\n";
            if (force_file.is_open()) {
                force_file << "Time step: " << t << "\n";
                force_file << "Fx = " << fx << "\n";
                force_file << "Fy = " << fy << "\n";
                force_file.close();
            } else {
                cerr << "Unable to open force output file." << endl;
            }
            
        }

    }
    double end_time = omp_get_wtime();

    std::cout << "Matrix inversion complete.\n";
    std::cout << "Execution time =  " << (end_time - start_time) << " seconds\n";
   return 0;
}

double summed_multiplication(const vector<double>& N, const vector<double>& xp) {
    double x = 0;
    for (int i = 0; i < 4; i++) {
        x += N[i] * xp[i];
    }
    return x;
}

vector<vector<double>> maping(const vector<double>& xcoord, const vector<double>& ycoord, 
                           double zita, double eta, double &determinent) {
    vector<double> N(4, 0.0);
    vector<double> Nx(4, 0.0), Ny(4, 0.0);
    vector<double> Nzita(4, 0.0), Neta(4, 0.0);

    Nzita[0] = -1*(1-eta)/4.0;
    Nzita[1] = (1-eta)/4.0;
    Nzita[2] = (1+eta)/4.0;
    Nzita[3] = -1*(1+eta)/4.0;

    Neta[0] = (zita - 1)/4.0;
    Neta[1] = (-zita - 1)/4.0;
    Neta[2] = (zita + 1)/4.0;
    Neta[3] = (1 - zita)/4.0;

    double xeita = summed_multiplication(Neta, xcoord);
    double yeita = summed_multiplication(Neta, ycoord);
    double xzita = summed_multiplication(Nzita, xcoord);
    double yzita = summed_multiplication(Nzita, ycoord);

    determinent = xzita*yeita - yzita*xeita;
    
    if (determinent <= 0) {
        cerr << "Error: Non-positive determinant detected: " << determinent << endl;
        exit(1);
    }

    N[0] = (1-zita)*(1-eta)/4;
    N[1] = (1+zita)*(1-eta)/4;
    N[2] = (1+zita)*(1+eta)/4;
    N[3] = (1-zita)*(1+eta)/4;

    for (int h = 0; h < 4; h++) {
        Nx[h] = (yeita*Nzita[h] - yzita*Neta[h]) / determinent;
        Ny[h] = (-xeita*Nzita[h] + xzita*Neta[h]) / determinent;
    }

    return {Ny, Nx, N};
}

void writeSolutionToVTK(const string& filename, 
    const vector<double>& x_coords, 
    const vector<double>& y_coords, 
    const vector<vector<int>>& connectivity, 
    const vector<double>& u, 
    const vector<double>& v,
    const vector<double>& p) {
    
    ofstream vtkfile(filename);
    if (!vtkfile.is_open()) {
        cerr << "Error: Could not open file " << filename << " for writing." << endl;
        return;
    }

    int NT = x_coords.size();
    int ET = connectivity.size();

    vtkfile << "# vtk DataFile Version 3.0\n";
    vtkfile << "Velocity and Pressure Field\n";
    vtkfile << "ASCII\n";
    vtkfile << "DATASET UNSTRUCTURED_GRID\n";

    // Write points
    vtkfile << "POINTS " << NT << " float\n";
    for (int i = 0; i < NT; ++i) {
        vtkfile << fixed << setprecision(6)
                << x_coords[i] << " " << y_coords[i] << " 0.0\n";
    }

    // Write cell connectivity
    vtkfile << "\nCELLS " << ET << " " << ET * 5 << "\n";
    for (const auto& conn : connectivity) {
        vtkfile << "4 " << conn[0] << " " << conn[1] << " "
                << conn[2] << " " << conn[3] << "\n";
    }

    // Define cell types
    vtkfile << "\nCELL_TYPES " << ET << "\n";
    for (int i = 0; i < ET; ++i) {
        vtkfile << "9\n";
    }

    // Write velocity field
    vtkfile << "\nPOINT_DATA " << NT << "\n";
    vtkfile << "VECTORS velocity float\n";
    for (int i = 0; i < NT; ++i) {
        vtkfile << u[i] << " " << v[i] << " 0.0\n";
    }

    // Write pressure field
    vtkfile << "\nSCALARS pressure float 1\n";
    vtkfile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < NT; ++i) {
        vtkfile << fixed << setprecision(6) << p[i] << "\n";
    }

    vtkfile.close();
    cout << "VTK file written to: " << filename << endl;
}