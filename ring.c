#include "mpi.h"
#include <stdio.h>


int main(int argc, char* argv[]) {
    
    MPI_Init(&argc, &argv);
    int rank, numRanks, received, sent;
    double startS, endS, totTimeS;
    //double startR, endR, totTimeR;
    startS = 0; endS = 0; totTimeS = 0;
    //startR = 0; endR = 0; totTimesR = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //printf("rank \n");
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    //printf("size \n");
    sent = 5;
    if (rank != 0) {
        MPI_Recv(&received, 1, MPI_INT, ((rank-1)%numRanks), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
        received = -1;
    }
        startS = MPI_Wtime();
        MPI_Send(&sent, 1, MPI_INT, ((rank+1)%numRanks), 0, MPI_COMM_WORLD);
        endS = MPI_Wtime();
        totTimeS = endS - startS;
    if (rank == 0) { //handles last process
        MPI_Recv(&received, 1, MPI_INT, numRanks-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } 
    printf("Process %d received %d from process %d \n", rank, received, ((numRanks+(rank-1))%numRanks));
    printf("The total time for sending the value %d from process %d to process %d was: %lf\n", received, rank, ((rank-1)%numRanks), totTimeS);
    MPI_Finalize();
    return 0;
}
