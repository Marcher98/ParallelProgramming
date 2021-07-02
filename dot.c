#include"mpi.h"
#include<stdio.h>
#include<stdlib.h>

/*
 * This method performs the dotproduct of two vectors using MPI_Scatterv
 */
void scatterV(int vectorSize, int rank, int ranks) {
    int countr; /* Counting results */
    int *vecA, *buffA; /* Declaring Vector A and buffer array used for MPI */
    int *vecB, *buffB; /* Declaring Vector B and buffer array used for MPI */
    int *result, *count; /* Declaring result array, and array for counting  */
    int sum = 0, res = 0; /* Variables for computation */
    int dcols, lcols; /* Distribution Column as well as the left columns */
    
    /* Allocating space to hold data... however you'd like to look at it .. fills aray*/
    if(rank == 0) {
        vecA=malloc(vectorSize*sizeof(int)); /* Initializing vecA with size (vectorSize*sizeof(int)) */
        vecB=malloc(vectorSize*sizeof(int)); /* Initializing vecB with size (vectorSize*sizeof(int)) */
        for(int i = 0; i < vectorSize; i++) { /* Iterating over vector size to fill vectorA and vectorB */
            vecA[i] = 5 + i;
            vecB[i] = vecA[i] + 500;
        }
    }
    
    /* Using MPI_Bcast to broadcast the datasize */
    MPI_Bcast(&vectorSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Calculate the distributed colums and left out columns */
    if(rank == 0) {
        result = malloc(ranks*sizeof(int));
        count = malloc(ranks*sizeof(int));
        dcols = vectorSize/ranks;
        lcols = vectorSize%ranks;
        for(int i = 0; i < ranks; i++) {
            count[i] = dcols;
        }
        for(int i = 0; i < lcols; i++) {
            count[i] = count[i] + 1;
        }
        
        /* This section of code calculates the addresses for scattering*/
        result[0] = 0;
        for(int i = 1; i < ranks; i++) {
            int a = 0;
            for(int j = 0; j < i; j++) {
                a = a + count[j];
            }
            result[i] = a;
        }
        
        /* this portion of the code sends countr to all ranks except rank 0*/
        for(int i = 1; i < ranks; i++) {
            countr = count[i];
            MPI_Send(&countr, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        countr = count[0];
    }
    
    /* Receive countr for all ranks not 0 */
    if(rank != 0) {
        MPI_Recv(&countr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    /* creates a buffer array to hold data during each process*/
    buffA=malloc(countr*sizeof(int));
    buffB=malloc(countr*sizeof(int));
    
    /* distribute stuff using scatterv :)*/
    MPI_Scatterv(vecA, count, result, MPI_INT, buffA, countr, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vecB, count, result, MPI_INT, buffB, countr, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Dot product computed here....*/
    for(int i = 0; i < countr; i++) {
        sum = sum + (buffA[i]*buffB[i]);
    }
    
    /* Use MPI_reduce to get information for output*/
    MPI_Reduce(&sum, &res, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    /* outputting all of the results....*/
    if(rank==0) {
        printf("The results from scatterv are: \n");
        printf("Product of vector A and Vector B : %d\n", res);
        /* Freeing memory.... */
        free(count);
        free(result);
        free(vecA);
        free(vecB);
    }
    free(buffA);
    free(buffB);
}

/*
 * This method performs the dotproduct of two vectors using MPI_Scatter
 */
void scatter(int vectorSize, int rank, int ranks) {
    int *result, *count;
    int sum=0, res=0;
    int countr;
    int *vecA, *buffA;
    int *vecB, *buffB;
    int dcols, lcols;
    
    /* Creation of space to hold data...*/
    if(rank == 0) {
        vecA = malloc(vectorSize*sizeof(int));
        vecB = malloc(vectorSize*sizeof(int));
        for(int i = 0; i < vectorSize; i++) {
            vecA[i] = 5 + i;
            vecB[i] = vecA[i] + 500;
        }
    }
    
    /* Broadcast the datasize */
    MPI_Bcast(&vectorSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Calculate the distributed colums and left out columns */
    if(rank == 0) {
        result = malloc(ranks*sizeof(int));
        count = malloc(ranks*sizeof(int));
        dcols = vectorSize/ranks;
        lcols = vectorSize%ranks;
        for(int i = 0; i < ranks; i++) {
            count[i] = dcols;
        }
        for(int i = 0; i < lcols; i++) {
            count[i] = count[i] + 1;
        }
        
        /* Caluclating the addresses for scattering i.e displacement values....*/
        result[0] = 0;
        for(int i = 1; i < ranks; i++) {
            int a = 0;
            for(int j = 0; j < i; j++) {
                a = a + count[j];
            }
            result[i] = a;
        }
        
        /* send the count of data to each proc......*/
        for(int i = 1; i < ranks; i++) {
            countr = count[i];
            MPI_Send(&countr, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        countr = count[0];
    }
    
    if(rank != 0) {
        MPI_Recv(&countr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    /* create buffer to hold elements at each proc....*/
    buffA=malloc(countr*sizeof(int));
    buffB=malloc(countr*sizeof(int));
    
    /* Distribute the data using Scatterv call....*/
    MPI_Scatterv(vecA, count, result, MPI_INT, buffA, countr, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vecB, count, result, MPI_INT, buffB, countr, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* computes dot product*/
    for(int i = 0; i < countr; i++) {
        sum= sum + (buffA[i]*buffB[i]);
    }
    
    /* gather all of the info for output*/
    MPI_Reduce(&sum, &res, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    /* outputting results */
    if(rank==0) {
        printf("\nThe results from scatter are: \n");
        printf("Product of vector A and Vector B : %d\n", res);
        /* Free the memory .......*/
        free(count);
        free(result);
        free(vecA);
        free(vecB);
    }
    free(buffA);
    free(buffB);
}

int main(int argc, char *argv[]) {
    int rank, ranks; /* current rank, and number of ranks (total) */
    double startv, endv, totTimev; /* variables used for timing scatterv */
    double start, end, totTime; /* variables used for timing scatter */
    
    /* Initialize MPI */
    MPI_Init(&argc,&argv);
    
    /* Gets number of ranks (total) */
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    
    /* Gets the current rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    /* Sets the size of the vector... (vectorSizex1 matrix if that helps imagining) */
    int vectorSize = 10;
    
    /* Timing scatterV */
    startv = MPI_Wtime();
    scatterV(vectorSize, rank, ranks);
    endv = MPI_Wtime();
    /* ***************************************************** */
    
    /* Timing scatter */
    start = MPI_Wtime();
    scatter(vectorSize, rank, ranks);
    end = MPI_Wtime();
    /* ***************************************************** */
    
    /* Calculates total time for Scatter V */
    totTimev = endv - startv;
    /* ***************************************************** */
    
    /* Calculates total time for scatter */
    totTime = end - start;
    /* ***************************************************** */
    
    /* Finalizing MPI */
    MPI_Finalize();
    
    /* Outputting the time calculated for scatter and scatter v with respective rank */
    printf("\nThe total time to complete scatterv on rank: %d with a vector size of: %d was: %lf", rank, vectorSize, totTimev);
    printf("\nThe total time to complete scatter on rank: %d with a vector size of: %d was: %lf", rank, vectorSize, totTime);
    
    /* End of Program */
    return 0;
    }
