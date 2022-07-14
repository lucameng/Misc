#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double T[10][10];
double Infsim_Arc(double p, double x){
	return 4 * pow((1 + pow((pow(x, p) / (1 - pow(x, p))), p-1)), 1.0/p);
}


double Romberg(double a, double b, double p, double error){
	double h, F; 
	int n, k;
	n = 1, k = 1;
	h = (b - a) / 2;
	F = 0;
	T[0][0] = h * (Infsim_Arc(p, a) + Infsim_Arc(p, b));
	
	while(1){
		for(int j = 1; j <= n; j++){
			F = F + Infsim_Arc(p, a + (2 * j - 1) * h);
		}
		
		T[k][0] = T[k-1][0] / 2 + h * F;
		
		for(int m = 1; m <= k; m++){
			T[k-m][m] = (pow(4,m) * T[k-m+1][m-1] - T[k-m][m-1]) / (pow(4,m) - 1);
		}
		
		int m = 1;
		h = h / 2;
		n = 2 * n;
		k ++;
		F = 0;
		
		if(fabs(T[0][k-1] - T[0][k-2]) < error) break;
	}
	return  T[0][k-1]; 
}


int main(){
	double p;
	scanf("%lf", &p); 
	double a = 0.0;
	double b = pow(2, -1.0/p);
	double error = 0.0000000001;
	printf("%.9f", Romberg(0, b, p, error));
	return 0;
}
