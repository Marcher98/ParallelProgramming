
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "myheader.h"

/*
*   This method allocates the correct amount of space for each array, the matrix (mxn), the vector(nx1), and then the product vector(nx1)
*/
void allocArrays(double** localA, double** localx, double** localy, int localm, int n, int localn) {
    int localok = 1;
    int ok;
    *localA = malloc(localm*n*sizeof(double));
    *localx = malloc(localn*sizeof(double));
    *localy = malloc(localm*sizeof(double));
    if (*localA == NULL || localx == NULL || localy == NULL) {
        localok = 0;
        MPI_Allreduce(&localok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }
}

/*
*   This method sets the size of the array/vector mp = m, np = n of mxn array and nx1 vectors
*/
void getDims(int* mp, int* localmp, int* np, int* localnp, int rank, int size) {
    int localok = 1;
    int ok;
    if (rank == 0) { //if main node..
        *mp = 8;
        *np = 8;
    }

    MPI_Bcast(mp, 1, MPI_INT, 0, MPI_COMM_WORLD); //broadcasts mp
    MPI_Bcast(np, 1, MPI_INT, 0, MPI_COMM_WORLD); //broadcasts np
    if ((*mp <= 0) || (*np <= 0) || (*mp%size != 0) || (*np%size != 0)) {
        localok = 0;
    }
    MPI_Allreduce(&localok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    *localmp = *mp/size;
    *localnp = *np/size;
}


/*
*   This method will read in a matrix using scatter
*/
void readMat(char prmpt[], double localA[], int m, int localm, int n, int rank) {
    double* temp = NULL;
    int localok = 1;
    int ok;
    if (rank == 0) {
        temp = malloc(m*n*sizeof(double));
        if (temp == NULL) {
            localok = 0;
        }
        MPI_Allreduce(&localok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        //printf("Enter the matrix %s\n", prmpt); -- used for user input
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                //scanf("%lf", &temp[i*n+j]); -- used for user input
                temp[i*n+j] = (i*j)+1;
            }
        }
        MPI_Scatter(temp, localm*n, MPI_DOUBLE, localA,localm*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(temp);
    } else {
        MPI_Allreduce(&localok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Scatter(temp, localm*n, MPI_DOUBLE, localA, localm*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void readVec(char prmpt[], double localvec[], int n, int localn, int rank) {
    double* vector = NULL;
    int localok = 1;
    int ok;
    if (rank == 0) {
        vector = malloc(n*sizeof(double));
        if (vector == NULL) {
            localok = 0;
        }
        MPI_Allreduce(&localok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        //printf("Enter the vector x %s\n", prmpt); -- used for user input
        for (int i = 0; i < n; i++) {
            //scanf("%lf", &vector[i]); -- used for user input
            vector[i] = (i*i)+1;
        }
        MPI_Scatter(vector, localn, MPI_DOUBLE, localvec, localn, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(vector);
    } else {
        MPI_Allreduce(&localok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Scatter(vector, localn, MPI_DOUBLE, localvec, localn, MPI_DOUBLE,0, MPI_COMM_WORLD);
    }
}

void printMat(char title[], double localvec[], int m, int localm, int n, int rank) {
    double* temp = NULL;
    int localok = 1;
    int ok;
    if (rank == 0) {
        temp = malloc(m*n*sizeof(double));
        if (temp == NULL) {
            localok = 0;
        }
        MPI_Allreduce(&localok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Gather(localvec, localm*n, MPI_DOUBLE, temp, localm*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        printf("The matrix %s looks like: \n", title);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%0.3f ", temp[i*n+j]);
            }
            printf("\n");
        }
        free(temp);
    } else {
        MPI_Allreduce(&localok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Gather(localvec, localm*n, MPI_DOUBLE, temp, localm*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void printVec(char title[], double localA[], int n, int localn, int rank) {
    double* temp = NULL;
    int localok = 1;
    int ok;
    if (rank == 0) {
        temp = malloc(n*sizeof(double));
        if (temp == NULL) {
            localok = 0;
        }
        MPI_Allreduce(&localok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Gather(localA, localn, MPI_DOUBLE, temp, localn, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        printf("The vector %s looks like: \n", title);
        for (int i = 0; i < n; i++) {
            printf("%0.3f ", temp[i]);
        }
        printf("\n");
        free(temp);
    } else {
        MPI_Allreduce(&localok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Gather(localA, localn, MPI_DOUBLE, temp, localn, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void mvp (double localA[], double localx[], double localy[], int localm, int n, int localn) {
    double* result = malloc(n*sizeof(double));
    int localok = 1;
    int ok;
    if (result == NULL) {
        localok = 0;
    }
    MPI_Allreduce(&localok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allgather(localx, localn, MPI_DOUBLE, result, localn, MPI_DOUBLE, MPI_COMM_WORLD);
    for (int i = 0; i < localm; i++) {
        localy[i] = 0.0;
        for (int j = 0; j < n; j++) {
            localy[i] = localy[i] + localA[i*n+j]*result[j];
        }
    }
    free(result);
}
int main(int argc, char *argv[]){
    double* localA;
    double* localX;
    double* localY;
    double startS, endS, totTimeS, startC, endC, totTimeC;
    startS = 0; endS = 0; totTimeS = 0;
    startC = 0; endC = 0; totTimeC = 0;
    int m, localm, n, localn, rank, numranks, localok;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    startS = MPI_Wtime(); //starts communication timing
    getDims(&m, &localm, &n, &localn, rank, numranks);

    allocArrays(&localA, &localX, &localY, localm, n, localn);

    readMat("A", localA, m, localm, n, rank);

    if (m < 10) {
        printMat("A", localA, m, localm, n ,rank);
    }

    readVec("x", localX, n, localn, rank);
    endS = MPI_Wtime(); //end of communication timing
    if (n < 10) {
        printVec("x", localX, n, localn, rank);
    }
    startC = MPI_Wtime(); //start of computation time
    mvp(localA, localX, localY, localm, n, localn);
    endC = MPI_Wtime(); //end of computation time

    if (n < 10) {
        printVec("y", localY, m, localm, rank);
    }
    
    totTimeS = endS - startS;
    totTimeC = endC - startC;

    printf("The total communication time was: %lf to compute the product vector with rank (%d) the time was: %lf\n", totTimeS, rank, totTimeC);
    
    
    MPI_Finalize();
    return 0;
}
