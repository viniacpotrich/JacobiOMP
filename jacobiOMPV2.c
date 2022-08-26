#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

/*----------------------------------------------------------*/
#define ERRO 1e-10

float calcErro(float *a, float *b, int tamA){
    float somaA = 0, somaB = 0, resultA = 0, resultB = 0;
    for(int i = 0; i < tamA; i++){
        somaA += pow(a[i], 2);
        somaB += pow(b[i], 2);
    }
    resultA = sqrt(somaA);
    resultB = sqrt(somaB);
    return resultA - resultB;
}

/*----------------------------------------------------------*/
void jacobi(int *A, float *b, float *x, int tamA, int nIteracoes, int nThreads){
    int id, i, j, inicio, fim, countIteracoes, marcador;
    float soma;
    float *xAtual, *xNovo;
    xAtual = (float *)malloc(tamA * sizeof(float));
    xNovo = (float *)malloc(tamA * sizeof(float));
    for (int i = 0; i<tamA; i++){
        xAtual[i] = 0;
        xNovo[i] = 0;
    }
    countIteracoes = 0;
    while (countIteracoes < nIteracoes && marcador != 1){
        #pragma omp parallel private(id, i, j, inicio, fim, soma)
        {
            id = omp_get_thread_num();
            inicio = id * tamA/nThreads;
            fim = (id + 1) * tamA/nThreads;
            //printf("id: %d inicio: %d fim: %d countI = %d numI %d\n", id, inicio, fim, countIteracoes,nIteracoes);
            for (i = inicio; i < fim; i++){
                soma = 0;
                for (j = 0; j < tamA; j++){
                    if (i != j){
                        soma += A[i*tamA+j] * xAtual[j];
                    }
                }
                xNovo[i] = (b[i] - soma)/A[i*tamA+i];
            }
            #pragma omp barrier
            if (id == 0){
                if ( fabs( calcErro(xAtual, xNovo, tamA) ) < ERRO){
                    for (i = 0; i < tamA; i++){
                        x[i] = xNovo[i];
                    }
                    marcador = 1;
                } else {
                    for (i = 0; i < tamA; i++){
                        xAtual[i] = xNovo[i];
                        x[i] = xAtual[i];
                    }
                }
                countIteracoes++;
                //printf("cont= %d no pai\n",countIteracoes);
            }
        }
    }
    printf("iteracoes: %d\n", countIteracoes);
}

void geraVetorMatriz(int *A, int tamA){
    for (int i = 0; i<tamA; i++){ // linha
        for (int j = 0; j<tamA; j++){ // coluna
            if ((i == 0 && j == 0) || (i == tamA-1 && j == tamA-1)){
                A[i*tamA+j] = 6;
            } else if (i == j){
                A[i*tamA+j] = 4;
            } else if (((j == i-1) || (j == i+1)) && i != 0 && i != tamA-1){
                A[i*tamA+j] = 1;
            } else {
                A[i*tamA+j] = 0;
            }
        }
    }
    return;
}

void escreveVetorMatriz(int *A, int tamLinha, int tamColuna){
    for (int i = 0; i<tamColuna; i++){ // linha
        for (int j = 0; j<tamLinha; j++){ // coluna
            printf("%lf ", A[i*tamLinha+j]);
        }
        printf("\n");
    }
    return;
}


void geraVetorResultado(float *b, int tamA){
    for (int i = 0; i < tamA; i++){
        //printf("i = %d\n",i);
        if ((i == 0) || (i == tamA-1)){
            b[i] = 0;
        } else if ((i == 1) || (i == tamA-2)){
            b[i] = 1;
        } else if((i == 2) || (i == tamA-3)){
            b[i] = 2;
        } else{
            b[i] = -6;
        }
    }
}

void escreveVetorResultado(float *b, int tamA){
    for (int i = 0; i < tamA; i++){
        printf("%lf ",b[i]);
    }
    printf("\n");
    return;
}

/*----------------------------------------------------------*/
int main(int argc, char **argv ){
    int tamA, nIteracoes, nThreads;
    double ti,tf=0;
	if ( argc != 4 ){
		printf("%s < Ordem da Matriz > < Max Iteracoes > < Threads >\n", argv[0]);
		exit(0);
	}
    tamA = atoi(argv[1]);
	nIteracoes = atoi(argv[2]);
	nThreads = atoi(argv[3]);
    int *A = (int *)malloc(tamA * tamA * sizeof(int));
    float *b = (float *)malloc(tamA * sizeof(float));
    geraVetorMatriz(A,tamA);
    //escreveVetorMatriz(A,tamA,tamA);
    geraVetorResultado(b,tamA);
    //escreveVetorResultado(b,tamA);

    float *x = (float *)malloc(tamA * sizeof(float));
    omp_set_num_threads(nThreads);
    ti = omp_get_wtime();
    jacobi(A, b, x, tamA, nIteracoes, nThreads);
    tf = omp_get_wtime();
    // printf("resultado\n");
    // for (int i = 0; i<tamA; i++){
    //     printf("%f\n", x[i]);
    // }
    printf("Tempo = %lf\n",tf-ti);
}
/*----------------------------------------------------------*/

