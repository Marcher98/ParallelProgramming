#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <mpi.h>

extern void matToImage(char* filename, int* mat, int* dims);

int main(int argc, char **argv){

    //init MPI
    int numranks,rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(comm, &numranks);
    MPI_Comm_rank(comm, &rank);
   
    //wtime
    double start, end, tot_time;

    int nx=12000;//7200; //cols //1 rank ~ 20sec
    int ny=8000;//4800; //rows
    int maxiter=255;
    int *mat;
    double area, i_c, r_c, i_z, r_z;
    double r_start, r_end, i_start, i_end;

    int numoutside=0;
    int iter;
    //vars for window size 3:2 ratio
    r_start = -2;
    r_end = 1;
    i_start = -1;
    i_end = 1;

    //MPI split
    int ny_sub = ny / numranks;
    int ny_start = ny_sub * rank;
    int ny_end = ny_start + (ny_sub);
    if(rank == (numranks-1)) {
        ny_end = ny;
    }
    mat=(int*)malloc(nx*ny*sizeof(int));
    int* sub_mat = (int*)malloc(nx*ny_sub*sizeof(int));

    start = MPI_Wtime();

    int counter = 0;
    for(int i=ny_start; i<ny_end; i++){
        #pragma omp parallel private(i_c, r_c, i_z, r_z, iter)
      {
        #pragma omp for reduction(+: numoutside) nowait schedule(dynamic)
        for(int j=0; j<nx; j++){    //cols
            i_c = i_start + i / (ny*1.0) * (i_end - i_start);
            r_c = r_start + j / (nx*1.0) * (r_end - r_start);
            i_z = i_c;
            r_z = r_c;
            iter=0;
            while(iter<maxiter){
                iter = iter+1;
                double r_t = r_z*r_z - i_z*i_z;
                double i_t = 2.0*r_z*i_z;
                i_z = i_t + i_c;
                r_z = r_t + r_c;
                if(r_z*r_z + i_z*i_z > 4){
                    numoutside = numoutside+1;
                    break;
                }
            }
            sub_mat[(i-ny_start)*nx+j] = iter;
        }
      }
    }
    end = MPI_Wtime();
    tot_time = end - start;

    printf("Rank: %d Time: %f\n",rank,tot_time);


    MPI_Barrier(comm);

    MPI_Gather(sub_mat,nx*ny_sub,MPI_INT,mat,nx*ny_sub,MPI_INT,0,comm);
    
    area = (r_end-r_start)*(i_end-i_start)*(1.0*nx*ny - numoutside) / (1.0*nx*ny);
    printf("Area of Mandelbrot set = %f\n",area);
    int dims[2] = {ny,nx};
    if(rank==0){
        matToImage("test.jpg",mat,dims);
    }//img_mandelbrot.jpg

    MPI_Finalize();
}
