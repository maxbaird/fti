/**
 *  @file   tokenRing.c
 *  @author Karol Sierocinski (ksiero@man.poznan.pl)
 *  @date   March, 2017
 *  @brief  FTI testing program.
 *
 *  First execution this program should be with fail flag = 1, because
 *  then FTI saves checkpoint and program stops after ITER_STOP iteration.
 *  Second execution must be with the same #defines and flag = 0 to
 *  properly recover data. It is important that FTI config file got
 *  keep_last_ckpt = 1.
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <fti.h>
#include <dirent.h>

#define ITERATIONS 599  //iterations for every level
#define ITER_CHECK 50   //every ITER_CHECK iterations make checkpoint
#define ITER_STOP 299    //end work after ITER_STOP iterations

#define WORK_DONE 0
#define CHECKPOINT_FAILED 1
#define RECOVERY_FAILED 2

#define VERIFY_SUCCESS 0
#define VERIFY_FAILED 1

/*-------------------------------------------------------------------------*/
/**
    @brief      Do work to makes checkpoints
    @param
    @param      fail        True if stop after ITER_STOP, false if resuming work
    @return     integer     WORK_DONE if successful.
 **/
/*-------------------------------------------------------------------------*/
int do_work(int* token, int world_rank, int world_size, int checkpoint_level, int fail) {
    typedef struct iteratiors {
        int i;
        int localIter;
    } cIters;
    cIters iters = {0,0};
    FTIT_type itersInfo;

    FTI_InitType(&itersInfo, 2*sizeof(int));
    FTI_Protect(1, &iters, 1, itersInfo);
    FTI_Protect(2, token, 1, FTI_INTG);
    if (FTI_Status() != 0 && fail == 0)
    {
        int res = FTI_Recover();
        if (res != 0) {
            printf("%d: FTI_Recover returned %d\n", world_rank, res);
            return RECOVERY_FAILED;
        }
    }
    if (world_rank == 0) {
        MPI_Request req;
        MPI_Isend(token, 1, MPI_INT, 1, 0, FTI_COMM_WORLD, &req);
        MPI_Isend(&(iters.i), 1, MPI_INT, 1, 0, FTI_COMM_WORLD, &req);
        MPI_Request_free(&req);
    }
    if (!fail && iters.localIter == 0) {
        printf("%d: Did not recovered.\n", world_rank);
        return RECOVERY_FAILED;
    }
    for (;iters.localIter < ITERATIONS/world_size + 1; iters.localIter++) {
        MPI_Barrier(FTI_COMM_WORLD);
        if (iters.localIter%(ITER_CHECK/world_size) == 0) {
            int res = FTI_Checkpoint(iters.localIter/(ITER_CHECK/world_size), checkpoint_level);
            if (res != FTI_DONE) {
                printf("%d: FTI_Checkpoint returned %d\n", world_rank, res);
                return CHECKPOINT_FAILED;
            }
        }
        if (fail && iters.localIter >= ITER_STOP/world_size) {
            printf("%d: Work stoped at i = %d.\n", world_rank, iters.i);
            break;
        }
        //ITERATIONS/world
        if (iters.i + world_size < ITERATIONS) {
            MPI_Recv(token, 1, MPI_INT, (world_rank + world_size - 1)%world_size, 0, FTI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&(iters.i), 1, MPI_INT, (world_rank + world_size - 1)%world_size, 0, FTI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("%d: Got token %d\n", world_rank, *token);
            (*token)++;
            iters.i++;
            MPI_Request req;
            MPI_Isend(token, 1, MPI_INT, (world_rank + 1)%world_size, 0, FTI_COMM_WORLD, &req);
            MPI_Isend(&(iters.i), 1, MPI_INT, (world_rank + 1)%world_size, 0, FTI_COMM_WORLD, &req);
            MPI_Request_free(&req);
            if (world_rank != 0) {
                *token = 0;
            }
        }
    }
    if (!fail) printf("%d: All work done.\n", world_rank);
    return WORK_DONE;
}


int init(char** argv, char* cnfgFile, int* checkpoint_level, int* fail) {
    int rtn = 0;    //return value
    if (argv[1] == NULL) {
        printf("Missing first parameter (config file).\n");
        rtn = 1;
    } else {
        cnfgFile = argv[1];
        //printf("Config file: %s\n", cnfgFile);
    }
    if (argv[2] == NULL) {
        printf("Missing second parameter (checkpoint level).\n");
        rtn = 1;
    } else {
        *checkpoint_level = atoi(argv[2]);
        //printf("Checkpoint level: %d\n", *checkpoint_level);
    }
    if (argv[3] == NULL) {
        printf("Missing third parameter (if fail).\n");
        rtn = 1;
    } else {
        *fail = atoi(argv[3]);
        //printf("Fail: %d\n", *fail);
    }
    return rtn;
}

int verify(int token, int world_rank, int world_size) {
    if (world_rank == 0) {
        if (token != ITERATIONS - ITERATIONS%world_size || token == 0) {
            printf("%d: Token == %d and ITERATIONSX = %d\n", world_rank, token, ITERATIONS - ITERATIONS%world_size);
            return VERIFY_FAILED;
        }
    } else {
        if (token != 0) {
            printf("%d: My token = %d", world_rank, token);
             return VERIFY_FAILED;
        }
    }
    return VERIFY_SUCCESS;
}

/*
    Prints:
        0 if everything is OK
        1 if calculation error
        2 if checkpoint failed
        3 if recovery failed
*/
int main(int argc, char** argv){
    char *cnfgFile;
    int checkpoint_level, fail;
    if (init(argv, cnfgFile, &checkpoint_level, &fail)) return 0;   //verify args

    MPI_Init(&argc, &argv);
    FTI_Init(argv[1], MPI_COMM_WORLD);
    int world_rank, world_size;
    MPI_Comm_rank(FTI_COMM_WORLD, &world_rank);
    MPI_Comm_size(FTI_COMM_WORLD, &world_size);

    int token = 0;
    MPI_Barrier(FTI_COMM_WORLD);
    int rtn; //return value

    rtn = do_work(&token, world_rank, world_size, checkpoint_level, fail);
    if (!rtn && !fail) rtn = verify(token, world_rank, world_size);
    FTI_Finalize();
    MPI_Finalize();
    return rtn;
}
