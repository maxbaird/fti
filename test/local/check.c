/**
 *  @file   check.c
 *  @author Kai Keller (kellekai@gmx.de)
 *  @date   June, 2017
 *  @brief  FTI testing program.
 *
 *	The program may test the correct behaviour for checkpoint
 *	and restart for all configurations. The recovered data is also
 *	tested upon correct data fields.
 *
 *	The program takes four arguments:
 *	  - arg1: FTI configuration file
 *	  - arg2: Interrupt yes/no (1/0)
 *	  - arg3: Checkpoint level (1, 2, 3, 4)
 *	  - arg4: different ckpt. sizes yes/no (1/0)
 *
 * If arg2 = 0, the program simulates a clean run of FTI:
 *    FTI_Init
 *    FTI_Protect
 *    if FTI_Status = 0
 *      FTI_Checkpoint
 *    else
 *      FTI_Recover
 *    FTI_Finalize
 *
 * If arg2 = 1, the program simulates an execution failure:
 *    FTI_Init
 *    FTI_Protect
 *    if FTI_Status = 0
 *      exit(10)
 *    else
 *      FTI_Recover
 *    FTI_Finalize
 *
 */

#include "mpi.h"
#include "fti.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#define N 100000
#define CNTRLD_EXIT 10
#define INIT_FAILED 20
#define RECOVERY_FAILED 30
#define PROTECT_FAILED 40
#define PROTECT_FAILED_AFTER_RAISE 50
#define DATA_CORRUPT 60
#define TEST_FAILED 70
#define KEEP 2
#define RESTART 1
#define INIT 0

/**
 * function prototypes
 */

/*-------------------------------------------------------------------------*/
/**
    @brief      Initialize test data
    @param      [out] A				Unit vector (1, 1, ....., 1)
    @param      [out] B				Random vector
    @param      [in] asize			Dimension

	Initializes A with 1's and B with random numbers r,  0 <= r <= 5.
	Dimension of both vectors is 'asize'
 **/
/*-------------------------------------------------------------------------*/
void init_arrays(double* A, double* B, size_t asize);

/*-------------------------------------------------------------------------*/
/**
    @brief      Multiplies components of A and B and stores result into A
    @param      [in/out] A			Unit vector (1, 1, ....., 1)
    @param      [in] B				Random vector
    @param      [in] asize			Dimension

    After function call, A equals B.
 **/
/*-------------------------------------------------------------------------*/
void vecmult(double* A, double* B, size_t asize);

/*-------------------------------------------------------------------------*/
/**
    @brief      Validifies the recovered data
    @param      [in] A			    A returned from vecmult
    @param      [in] B_chk			POSIX Backup of B
    @param      [in] asize			Dimension
    @return     integer             0 if successful, -1 else.

    Checks entry for entry if A equals the POSIX Backup of B, B_chk, from
    the preceding execution. This function must be called after the call to
    vecmult(A, B, asize).
 **/
/*-------------------------------------------------------------------------*/
int validify(double* A, double* B_chk, size_t asize);

/*-------------------------------------------------------------------------*/
/**
    @brief      Writes 'B' and 'asize' to file, using POSIX fwrite.
    @param      [in] B              Random array B from init_array call
    @param      [in] asize			Dimension
    @param      [in] rank           FTI application rank
 **/
/*-------------------------------------------------------------------------*/
int write_data(double* B, size_t* asize, int rank);

/*-------------------------------------------------------------------------*/
/**
    @brief      Recovers 'B' and 'asize' to 'B_chk' and 'asize_chk' from file,
                using POSIX fread.
    @param      [out] B_chk         B backup
    @param      [out] asize_chk     Dimension backup
    @param      [in] rank           FTI application rank
    @param      [in] asize			Dimension
    @return     integer             0 if successful, -1 else.

    Before recovering B, the function checks if 'asize_chk' equals 'asize',
    to prevent SIGSEGV. If not 'asize_chk' = 'asize' it returns -1.
 **/
/*-------------------------------------------------------------------------*/
int read_data(double* B_chk, size_t* asize_chk, int rank, size_t asize);

/**
 * main
 */

int main(int argc, char* argv[]) {

    int FTI_APP_RANK, MPI_RANK; 
    unsigned char parity, crash, level, state, diff_sizes, var_size;
    int result, tmp, success = 1, perr = 0;
    double *A, *B, *B_chk;
    char *errormsg;

    size_t asize, asize_chk;
    
    srand(time(NULL));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_RANK);
    
    if (argc < 6) {
        if( MPI_RANK == 0 ) {
            fprintf(stderr, "\n\t usage: check.exe configFile isCrash ckptLevel isDiffSizes isVarSizes\n\n");
            exit(TEST_FAILED);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);    

// INITIALIZE FTI
    perr = FTI_Init(argv[1], MPI_COMM_WORLD);
    if (perr != 0) {
        perr = errno;
        errormsg = strerror(perr);
        perr = 0;
        fprintf(stderr,"[ERROR rank - %i] FTI failed to initialize :: %s\n", MPI_RANK,  errormsg);
        MPI_Abort(MPI_COMM_WORLD, INIT_FAILED);
    }     
    
    MPI_Comm_rank(FTI_COMM_WORLD, &FTI_APP_RANK);

// REQUEST PARAMETER
    crash = atoi(argv[2]);
    level = atoi(argv[3]);
    diff_sizes = atoi(argv[4]);
    var_size = atoi(argv[5]);
    

// SET ARRAY LENGTHS FOR EACH RANK
    asize = N;

    if (diff_sizes) {
        parity = FTI_APP_RANK%7;

        switch (parity) {

            case 0:
                asize = N;
                break;

            case 1:
                asize = 2*N;
                break;

            case 2:
                asize = 3*N;
                break;

            case 3:
                asize = 4*N;
                break;

            case 4:
                asize = 5*N;
                break;

            case 5:
                asize = 6*N;
                break;

            case 6:
                asize = 7*N;
                break;

        }
    }

    A = (double*) malloc(asize*sizeof(double));
    B = (double*) malloc(asize*sizeof(double));

// PROTECT THE VARIABLES
    perr = FTI_Protect(0, A, asize, FTI_DBLE);
    if (perr != 0) {
        perr = errno;
        errormsg = strerror(perr);
        perr = 0;
        fprintf(stderr,"[ERROR rank - %i] FTI failed to protect a variable :: %s\n", FTI_APP_RANK,  errormsg);
        MPI_Abort(MPI_COMM_WORLD, PROTECT_FAILED);
    }     
    perr = FTI_Protect(1, B, asize, FTI_DBLE);
    if (perr != 0) {
        perr = errno;
        errormsg = strerror(perr);
        perr = 0;
        fprintf(stderr,"[ERROR rank - %i] FTI failed to protect a variable :: %s\n", FTI_APP_RANK,  errormsg);
        MPI_Abort(MPI_COMM_WORLD, PROTECT_FAILED);
    }     
    perr = FTI_Protect(2, &asize, 1, FTI_INTG);
    if (perr != 0) {
        perr = errno;
        errormsg = strerror(perr);
        perr = 0;
        fprintf(stderr,"[ERROR rank - %i] FTI failed to protect a variable :: %s\n", FTI_APP_RANK,  errormsg);
        MPI_Abort(MPI_COMM_WORLD, PROTECT_FAILED);
    }     

    state = FTI_Status();

// IF FTI STARTS CLEAN
    if (state == INIT) {
        if (var_size == 1) {
            asize *= 2; 
            A = (double*) malloc(asize*sizeof(double)); 
            B = (double*) malloc(asize*sizeof(double));
            FTI_Protect(0, A, asize, FTI_DBLE);
            if (perr != 0) {
                perr = errno;
                errormsg = strerror(perr);
                perr = 0;
                fprintf(stderr,"[ERROR rank - %i] FTI failed to protect a variable after increased the size :: %s\n", FTI_APP_RANK,  errormsg);
                MPI_Abort(MPI_COMM_WORLD, PROTECT_FAILED_AFTER_RAISE);
            }     
            FTI_Protect(1, B, asize, FTI_DBLE);
            if (perr != 0) {
                perr = errno;
                errormsg = strerror(perr);
                perr = 0;
                fprintf(stderr,"[ERROR rank - %i] FTI failed to protect a variable after increased the size :: %s\n", FTI_APP_RANK,  errormsg);
                MPI_Abort(MPI_COMM_WORLD, PROTECT_FAILED_AFTER_RAISE);
            }     
        }
        init_arrays(A, B, asize);
        write_data(B, &asize, FTI_APP_RANK);
        MPI_Barrier(FTI_COMM_WORLD);
        FTI_Checkpoint(1,level);
        sleep(5);
        if (crash && FTI_APP_RANK == 0) {
            exit(CNTRLD_EXIT);
        }
    }

// IF FTI STARTS AFTER CRASH
    if ( state == RESTART || state == KEEP ) {
        if (var_size == 1) {
            asize = FTI_GetStoredSize(0)/sizeof(double);
            A = (double*) FTI_Realloc(0, A);
            B = (double*) FTI_Realloc(1, B);
        }
        result = FTI_Recover();
        if (result != FTI_SCES) {
            exit(RECOVERY_FAILED);
        }
        B_chk = (double*) malloc(asize*sizeof(double));
        result = read_data(B_chk, &asize_chk, FTI_APP_RANK, asize);
        MPI_Barrier(FTI_COMM_WORLD);
        if (result != 0) {
            exit(DATA_CORRUPT);
        }
    }

    /*
     * on INIT, B is initialized randomly
     * on RESTART or KEEP, B is recovered and must be equal to B_chk
     */

    vecmult(A, B, asize);

    if (state == RESTART || state == KEEP) {
        result = validify(A, B_chk, asize);
        result += (asize_chk == asize) ? 0 : -1;
        MPI_Allreduce(&result, &tmp, 1, MPI_INT, MPI_SUM, FTI_COMM_WORLD);
        result = tmp;
        free(B_chk);
    }

    free(A);
    free(B);

    if (FTI_APP_RANK == 0 && (state == RESTART || state == KEEP)) {
        if (result == 0) {
            printf("[SUCCESSFUL]\n");
        } else {
            printf("[NOT SUCCESSFUL]\n");
            success=0;
        }
    }

    MPI_Barrier(FTI_COMM_WORLD);
    FTI_Finalize();
    MPI_Finalize();

    if (success == 1)
        return 0;
    else
        exit(DATA_CORRUPT);

}

/**
 * function definitions
 */

void init_arrays(double* A, double* B, size_t asize) {
    int i;
    double r;
    for (i = 0; i< asize; i++) {
        A[i] = 1.0;
        B[i] = ((double)rand()/RAND_MAX)*5.0;
    }
}

void vecmult(double* A, double* B, size_t asize) {
    int i;
    for (i=0; i<asize; i++) {
        A[i] = A[i]*B[i];
    }
}

int validify(double* A, double* B_chk, size_t asize) {
    int i;
    for (i=0; i<asize; i++) {
        if (A[i] != B_chk[i]){
            return -1;
        }
    }
    return 0;
}

int write_data(double* B, size_t *asize, int rank) {
    char str[256];
    int perr = 0;
    char *errormsg;
    sprintf(str, "chk/check-%i.tst", rank);
    FILE* f = fopen(str, "wb");
    if (f == NULL) {
        perr = errno;
        errormsg = strerror(perr);
        perr = 0;
        fprintf(stderr,"[ERROR rank - %i] Could not create test data file '%s' :: %s\n", rank, str, errormsg);
        MPI_Abort(MPI_COMM_WORLD, TEST_FAILED);
    }

    size_t written = 0;
    double *ptr;

    perr = fwrite( (void*) asize, sizeof(size_t), 1, f);
    if (perr == 0) {
        perr = errno;
        errormsg = strerror(perr);
        perr = 0;
        fclose(f);
        fprintf(stderr,"[ERROR rank - %i] Could not write test data :: %s\n", rank,  errormsg);
        MPI_Abort(MPI_COMM_WORLD, TEST_FAILED);
    }     

    ptr = B;
    while ( written < *asize ) {
        written += fwrite( (void*) ptr, sizeof(double), (*asize), f);
        if (written == 0) {
            perr = errno;
            errormsg = strerror(perr);
            perr = 0;
            fclose(f);
            fprintf(stderr,"[ERROR rank - %i] Could not write test data :: %s\n", rank,  errormsg);
            MPI_Abort(MPI_COMM_WORLD, TEST_FAILED);
        }     
        ptr += written;
    }

    fclose(f);

    return 0;
}

int read_data(double* B_chk, size_t *asize_chk, int rank, size_t asize) {
    char str[256];
    int perr = 0;
    char *errormsg;
    sprintf(str, "chk/check-%i.tst", rank);
    FILE* f = fopen(str, "rb");
    if (f == NULL) {
        perr = errno;
        errormsg = strerror(perr);
        perr = 0;
        fprintf(stderr,"[ERROR rank - %i] Could not open test data file '%s' :: %s\n", rank, str, errormsg);
        MPI_Abort(MPI_COMM_WORLD, TEST_FAILED);
    }
    size_t read = 0;
    double *ptr;

    perr = fread( (void*) asize_chk, sizeof(size_t), 1, f);
    if (perr == 0) {
        perr = errno;
        errormsg = strerror(perr);
        perr = 0;
        fclose(f);
        fprintf(stderr,"[ERROR rank - %i] Could not read test data :: %s\n", rank,  errormsg);
        MPI_Abort(MPI_COMM_WORLD, TEST_FAILED);
    }     
    if ((*asize_chk) != asize) {
        printf("[ERROR -%i] : wrong dimension 'asize' -- asize: %zd, asize_chk: %zd\n", rank, asize, *asize_chk);
        fflush(stdout);
        return -1;
    }

    ptr = B_chk;
    while ( read < *asize_chk ) {
        read += fread( (void*) ptr, sizeof(double), (*asize_chk), f);
        if (read == 0) {
            perr = errno;
            errormsg = strerror(perr);
            perr = 0;
            fclose(f);
            fprintf(stderr,"[ERROR rank - %i] Could not read test data :: %s\n", rank,  errormsg);
            MPI_Abort(MPI_COMM_WORLD, TEST_FAILED);
        }     
        ptr += read;

    }

    fclose(f);

    return 0;
}
