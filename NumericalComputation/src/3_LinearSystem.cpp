#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#define MAX 1024

int Hadamard[MAX][MAX], D[MAX];
double L[MAX][MAX];

int factor_h(int i, int j){     //Determine the value of each element
    long k, temp, result = 0;
    temp = i & j;
    for (k = 0; k < 32; k ++){
        result = result + (temp >> k) & 1;
    }
    if (result % 2 == 0){
        return 1;
    }
    else{
        return -1;
    }
}

void hadamard_generator(int k){     //Generate Hadamard matrix
    int n = pow(2, k);
    for (int i = 0; i < n; i ++)
    {
        for (int j = 0; j < n; j ++)
        {
            Hadamard[i][j] = factor_h(i, j);
        }
    }
}

void matrix_decomp(int A[MAX][MAX], int n){     //decompose matrix by LDL^T and solve the value of d_i
	double T[MAX];
	D[0] = A[0][0];
	for(int j = 1; j < n; j ++){
		for(int k = 0; k < j; k ++){
			T[k] = A[k][j];
			for(int i = 0; i < k; i ++){
			    T[k] -= L[k][i] * T[i]; 
			} 			
			L[j][k] = T[k] / (D[k] * 1.0);
		}
		D[j] = A[j][j];
		for(int i = 0; i < j; i ++){
			D[j] -= L[j][i] * T[i];
		}
	}
}

int main()
{
    int k;
    scanf("%d", &k);
    hadamard_generator(k);
    int n = pow(2, k);

    for(int i = 0; i < n; i ++){     //initialize
        D[i]=0.0;
        for(int j=0;j<n;j++){
            L[i][j] = 0.0;
        }
        L[i][i]=1.0;
    }

    matrix_decomp(Hadamard, n);

    for(int i = 0; i < n; ){
		printf("%d ", D[i]);
        if((++ i) % 8 == 0) 
            printf("\n");
	}
        
}

