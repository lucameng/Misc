#include <stdio.h>
#include <math.h>
#include <string.h>
#define MAX 51

double omega = 1;
double eps = 0.00000000001;
double mat_A[MAX][MAX];
double b[MAX];
double p[MAX];
double p_last[MAX];


// Matrix Assignment
void matrix_assign(double matrix[MAX][MAX], int n, double alpha){
    // To initialize the matrix
    for(int i = 0; i < MAX; i ++){
        for(int j = 0; j < MAX; j ++)
            matrix[i][j] = 0;
        }
    matrix[0][0] = 1;
    matrix[n - 1][n - 1] = 1;
    //assignments
    for(int i = 1; i < n - 1; i ++){
        for(int j = 1; j < n - 1; j ++){
            if(i == j){
                matrix[i][i] = 1;
                matrix[i][i - 1] = - alpha;
                matrix[i][i + 1] = alpha - 1;
            }
        }
    }
}


void print_matrix(double matrix[MAX][MAX], int n){
    for(int i = 0; i < n; i ++){
        for(int j = 0; j < n; j ++){
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

void interation(double matrix[MAX][MAX], double x[MAX], double b[MAX], int n, double omega){
    for(int i = 0; i < n; i ++){
        p_last[i] = x[i];
    }
    for(int i = 0; i < n; i ++){      
        x[i] = omega * b[i];
        for(int j = 0; j < i; j ++){
            x[i] = x[i] - omega * matrix[i][j] * x[j];
        }
        x[i] = x[i] + (1.0 - omega) * matrix[i][i] * p_last[i];
        for(int j = i + 1; j < n; j ++){
            x[i] = x[i] - omega * matrix[i][j] * p_last[j];
        }
        x[i] = x[i] / matrix[i][i];
    }    
}

double L_inf_norm(double a[MAX], double b[MAX], int n){
    double max = 0.0;
    for(int i = 0; i < n; i ++){
        if(abs(a[i] - b[i]) > max)
            max = abs(a[i] - b[i]);
    }
    return max * n;
}


int main(){
    double alpha = 0;   
    int dim = 0;
    scanf("%lf %d", &alpha, &dim); 
    dim += 1;
    matrix_assign(mat_A, dim, alpha);
    mat_A[0][0] = mat_A[dim - 1][dim - 1] = 1;

    // Initialize all vectors
    memset(b, 0, sizeof(b));
    memset(p, 0, sizeof(p));
    memset(p_last, 0, sizeof(p_last));
    b[0] = 1;
    b[dim - 1] = 0;

    int N = 10000000;   //Total Number of Iterations
    int k = 0;
    while(k < N){
        interation(mat_A, p, b, dim, omega);
        k ++;
        if(L_inf_norm(p, p_last, dim) < eps){
            for(int i = 1; i < dim - 1; i ++){
                printf("%.10f ", p[i]);
                if(i % 5 == 0) printf("\n");
            }
            break;
        }
    }
    return 0;
}
