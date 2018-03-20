/**
 *  Copyright (c) 2017 Leonardo A. Bautista-Gomez
 *  All rights reserved
 *
 *  FTI - A multi-level checkpointing library for C/C++/Fortran applications
 *
 *  Revision 1.0 : Fault Tolerance Interface (FTI)
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its contributors
 *  may be used to endorse or promote products derived from this software without
 *  specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  @file   diff-checkpoint.c
 *  @date   February, 2018
 *  @brief  Differential checkpointing routines.
 */

#define _DEFAULT_SOURCE
#define _BSD_SOURCE

#include "diff-checkpoint.h"

/**                                                                                     */
/** Static Global Variables                                                             */

static FTI_ADDRVAL          FTI_PageSize;       /**< memory page size                   */
static FTI_ADDRVAL          FTI_PageMask;       /**< memory page mask                   */

static FTIT_DataDiffInfoSignal    FTI_SignalDiffInfo;   /**< container for diff of datasets     */
static FTIT_DataDiffInfoHash      FTI_HashDiffInfo;   /**< container for diff of datasets     */

/** File Local Variables                                                                */

static bool enableDiffCkpt;
static int diffMode;

static struct sigaction     FTI_SigAction;       /**< sigaction meta data               */
static struct sigaction     OLD_SigAction;       /**< previous sigaction meta data      */

/** Function Definitions                                                                */

int FTI_InitDiffCkpt( FTIT_configuration* FTI_Conf, FTIT_execution* FTI_Exec, FTIT_dataset* FTI_Data )
{
    enableDiffCkpt = FTI_Conf->enableDiffCkpt;
    diffMode = FTI_Conf->diffMode;
    if( enableDiffCkpt && FTI_Conf->diffMode == 0 ) {
        FTI_HashDiffInfo.dataDiff = NULL;
        FTI_HashDiffInfo.nbProtVar = 0;
        return FTI_SCES;
    }
    if( enableDiffCkpt && FTI_Conf->diffMode == 1 ) {
        // get page mask
        FTI_PageSize = (FTI_ADDRVAL) sysconf(_SC_PAGESIZE);
        FTI_PageMask = ~((FTI_ADDRVAL)0x0);
        FTI_ADDRVAL tail = (FTI_ADDRVAL)0x1;
        for(; tail!=FTI_PageSize; FTI_PageMask<<=1, tail<<=1); 
        // init data diff structure
        FTI_SignalDiffInfo.dataDiff = NULL;
        FTI_SignalDiffInfo.nbProtVar = 0;
        // register signal handler
        return FTI_RegisterSigHandler();
    } else {
        return FTI_SCES;
    }
}

void printReport() {
    long num_unchanged = 0;
    long num_prot = 0; 
    int i,j;
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i) {
        num_prot += FTI_SignalDiffInfo.dataDiff[i].totalSize;
        for(j=0; j<FTI_SignalDiffInfo.dataDiff[i].rangeCnt; ++j) {
            num_unchanged += FTI_SignalDiffInfo.dataDiff[i].ranges[j].size;
        }
    }
    num_prot /= FTI_PageSize;
    num_unchanged /= FTI_PageSize;

    printf(
            "Diff Ckpt Summary\n"
            "-------------------------------------\n"
            "number of pages protected:       %lu\n"
            "number of pages changed:         %lu\n",
            num_prot, num_unchanged);
    fflush(stdout);
    
}

int FTI_FinalizeDiffCkpt(){
    int res = 0;
    if( enableDiffCkpt ) {
        if( diffMode == 1 ) {
            //printReport();
            res += FTI_RemoveSigHandler();
            res += FTI_RemoveProtections();
            res += FTI_FreeDiffCkptStructs();
        }
    }
    return ( res == 0 ) ? FTI_SCES : FTI_NSCS;
}

int FTI_FreeDiffCkptStructs() {
    int i;
    
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i) {
        if(FTI_SignalDiffInfo.dataDiff[i].ranges != NULL) {
            free(FTI_SignalDiffInfo.dataDiff[i].ranges);
        }
    }
    free(FTI_SignalDiffInfo.dataDiff);

    return FTI_SCES;
}

int FTI_RemoveProtections() {
    int i,j;
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i){
        for(j=0; j<FTI_SignalDiffInfo.dataDiff[i].rangeCnt; ++j) {
            FTI_ADDRPTR ptr = (FTI_ADDRPTR) (FTI_SignalDiffInfo.dataDiff[i].basePtr + FTI_SignalDiffInfo.dataDiff[i].ranges[j].offset);
            long size = (long) FTI_SignalDiffInfo.dataDiff[i].ranges[j].size;
            if ( mprotect( ptr, size, PROT_READ|PROT_WRITE ) == -1 ) {
                // ENOMEM is return e.g. if allocation was already freed, which will (hopefully) 
                // always be the case if FTI_Finalize() is called at the end and the buffer was allocated dynamically
                if ( errno != ENOMEM ) {
                    FTI_Print( "FTI was unable to restore the data access", FTI_EROR );
                    return FTI_NSCS;
                }
                errno = 0;
            }
        }
    }
    return FTI_SCES;
}

int FTI_RemoveSigHandler() 
{ 
    if( sigaction(SIGSEGV, &OLD_SigAction, NULL) == -1 ){
        FTI_Print( "FTI was unable to restore the default signal handler", FTI_EROR );
        errno = 0;
        return FTI_NSCS;
    }
    return FTI_SCES;
}

/*
    TODO: It might be good to remove all mprotect for all pages if a SIGSEGV gets raised
    which does not belong to the change-log mechanism.
*/

int FTI_RegisterSigHandler() 
{ 
    // SA_SIGINFO -> flag to allow detailed info about signal
    FTI_SigAction.sa_flags = SA_SIGINFO;
    sigemptyset(&FTI_SigAction.sa_mask);
    FTI_SigAction.sa_sigaction = FTI_SigHandler;
    if( sigaction(SIGSEGV, &FTI_SigAction, &OLD_SigAction) == -1 ){
        FTI_Print( "FTI was unable to register the signal handler", FTI_EROR );
        errno = 0;
        return FTI_NSCS;
    }
    return FTI_SCES;
}

void FTI_SigHandler( int signum, siginfo_t* info, void* ucontext ) 
{
    char strdbg[FTI_BUFS];

    if ( signum == SIGSEGV ) {
        if( FTI_isValidRequest( (FTI_ADDRVAL)info->si_addr ) ){
            
            // debug information
            snprintf( strdbg, FTI_BUFS, "FTI_DIFFCKPT: 'FTI_SigHandler' - SIGSEGV signal was raised at address %p\n", info->si_addr );
            FTI_Print( strdbg, FTI_DBUG );
            
            // remove protection from page
            if ( mprotect( (FTI_ADDRPTR)(((FTI_ADDRVAL)info->si_addr) & FTI_PageMask), FTI_PageSize, PROT_READ|PROT_WRITE ) == -1) {
                FTI_Print( "FTI was unable to register the signal handler", FTI_EROR );
                errno = 0;
                if( sigaction(SIGSEGV, &OLD_SigAction, NULL) == -1 ){
                    FTI_Print( "FTI was unable to restore the parent signal handler", FTI_EROR );
                    errno = 0;
                }
            }
            
            // register page as dirty
            FTI_ExcludePage( (FTI_ADDRVAL)info->si_addr );
        
        } else {
            /*
                NOTICE: tested that this works also if the application that leverages FTI uses signal() and NOT sigaction(). 
                I.e. the handler registered by signal from the application is called for the case that the SIGSEGV was raised from 
                an address outside of the FTI protected region.
                TODO: However, needs to be tested with applications that use signal handling.
            */

            // forward to default handler and raise signal again
            if( sigaction(SIGSEGV, &OLD_SigAction, NULL) == -1 ){
                FTI_Print( "FTI was unable to restore the parent handler", FTI_EROR );
                errno = 0;
            }
            raise(SIGSEGV);
            
            // if returns from old signal handler (which it shouldn't), register FTI handler again.
            // TODO since we are talking about seg faults, we might not attempt to register our handler again here.
            if( sigaction(SIGSEGV, &FTI_SigAction, &OLD_SigAction) == -1 ){
                FTI_Print( "FTI was unable to register the signal handler", FTI_EROR );
                errno = 0;
            }
        }
    }
}

int FTI_ExcludePage( FTI_ADDRVAL addr ) {
    bool found; 
    FTI_ADDRVAL page = addr & FTI_PageMask;
    int idx;
    long pos;
    if( FTI_GetRangeIndices( page, &idx, &pos) == FTI_NSCS ) {
        return FTI_NSCS;
    }
    // swap array elements i -> i+1 for i > pos and increase counter
    FTI_ADDRVAL base = FTI_SignalDiffInfo.dataDiff[idx].basePtr;
    FTI_ADDRVAL offset = FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].offset;
    FTI_ADDRVAL end = base + offset + FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].size;
    // update values
    FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].size = page - (base + offset);
    if ( FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].size == 0 ) {
        FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].offset = page - base + FTI_PageSize;
        FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].size = end - (page + FTI_PageSize);
        if ( FTI_SignalDiffInfo.dataDiff[idx].ranges[pos].size == 0 ) {
            FTI_ShiftPageItemsLeft( idx, pos );
        }
    } else {
        FTI_ShiftPageItemsRight( idx, pos );
        FTI_SignalDiffInfo.dataDiff[idx].ranges[pos+1].offset = page - base + FTI_PageSize;
        FTI_SignalDiffInfo.dataDiff[idx].ranges[pos+1].size = end - (page + FTI_PageSize);
    }

    // add dirty page to buffer
}

int FTI_GetRangeIndices( FTI_ADDRVAL page, int* idx, long* pos)
{
    // binary search for page
    int i;
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i){
        bool found = false;
        long LOW = 0;
        long UP = FTI_SignalDiffInfo.dataDiff[i].rangeCnt - 1;
        long MID = (LOW+UP)/2; 
        if ( FTI_RangeCmpPage(i, MID, page) == 0 ) {
            found = true;
        }
        while( LOW < UP ) {
            int cmp = FTI_RangeCmpPage(i, MID, page);
            // page is in first half
            if( cmp < 0 ) {
                UP = MID - 1;
            // page is in second half
            } else if ( cmp > 0 ) {
                LOW = MID + 1;
            }
            MID = (LOW+UP)/2;
            if ( FTI_RangeCmpPage(i, MID, page) == 0 ) {
                found = true;
                break;
            }
        }
        if ( found ) {
            *idx = i;
            *pos = MID;
            return FTI_SCES;
        }
    }
    return FTI_NSCS;
}

int FTI_RangeCmpPage(int idx, long idr, FTI_ADDRVAL page) {
    FTI_ADDRVAL base = FTI_SignalDiffInfo.dataDiff[idx].basePtr;
    FTI_ADDRVAL size = FTI_SignalDiffInfo.dataDiff[idx].ranges[idr].size;
    FTI_ADDRVAL first = FTI_SignalDiffInfo.dataDiff[idx].ranges[idr].offset + base;
    FTI_ADDRVAL last = FTI_SignalDiffInfo.dataDiff[idx].ranges[idr].offset + base + size - FTI_PageSize;
    if( page < first ) {
        return -1;
    } else if ( page > last ) {
        return 1;
    } else if ( (page >= first) && (page <= last) ) {
        return 0;
    }
}

int FTI_ShiftPageItemsLeft( int idx, long pos ) {
    // decrease array size by 1 and decrease counter and return if at the end of the array
    if ( pos == FTI_SignalDiffInfo.dataDiff[idx].rangeCnt - 1 ) {
        if ( pos == 0 ) {
            --FTI_SignalDiffInfo.dataDiff[idx].rangeCnt;
            return FTI_SCES;
        }
        FTI_SignalDiffInfo.dataDiff[idx].ranges = (FTIT_DataRange*) realloc(FTI_SignalDiffInfo.dataDiff[idx].ranges, (--FTI_SignalDiffInfo.dataDiff[idx].rangeCnt) * sizeof(FTIT_DataRange));
        assert( FTI_SignalDiffInfo.dataDiff[idx].ranges != NULL );
        return FTI_SCES;
    }
    long i;
    // shift elements of array starting at the end
    for(i=FTI_SignalDiffInfo.dataDiff[idx].rangeCnt - 1; i>pos; --i) {
        memcpy( &(FTI_SignalDiffInfo.dataDiff[idx].ranges[i-1]), &(FTI_SignalDiffInfo.dataDiff[idx].ranges[i]), sizeof(FTIT_DataRange));
    }
    FTI_SignalDiffInfo.dataDiff[idx].ranges = (FTIT_DataRange*) realloc(FTI_SignalDiffInfo.dataDiff[idx].ranges, (--FTI_SignalDiffInfo.dataDiff[idx].rangeCnt) * sizeof(FTIT_DataRange));
    assert( FTI_SignalDiffInfo.dataDiff[idx].ranges != NULL );
    return FTI_SCES;
}

int FTI_ShiftPageItemsRight( int idx, long pos ) {
    // increase array size by 1 and increase counter
    FTI_SignalDiffInfo.dataDiff[idx].ranges = (FTIT_DataRange*) realloc(FTI_SignalDiffInfo.dataDiff[idx].ranges, (++FTI_SignalDiffInfo.dataDiff[idx].rangeCnt) * sizeof(FTIT_DataRange));
    assert( FTI_SignalDiffInfo.dataDiff[idx].ranges != NULL );
    
    long i;
    // shift elements of array starting at the end
    assert(FTI_SignalDiffInfo.dataDiff[idx].rangeCnt > 0);
    for(i=FTI_SignalDiffInfo.dataDiff[idx].rangeCnt-1; i>(pos+1); --i) {
        memcpy( &(FTI_SignalDiffInfo.dataDiff[idx].ranges[i]), &(FTI_SignalDiffInfo.dataDiff[idx].ranges[i-1]), sizeof(FTIT_DataRange));
    }
}

int FTI_RegisterProtections(int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec) 
{   
    if( diffMode == 0 ) {
        return FTI_GenerateHashBlocks( idx, FTI_Data, FTI_Exec );
    } else if ( diffMode == 1 ) {
        return FTI_ProtectPages ( idx, FTI_Data, FTI_Exec );
    } else {
        return FTI_SCES;
    }

}

int FTI_UpdateProtections(int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec) 
{   
    if( diffMode == 0 ) {
        return FTI_UpdateHashBlocks( idx, FTI_Data, FTI_Exec );
    //} else if ( diffMode == 1 ) {
    //    return FTI_ProtectPages ( idx, FTI_Data, FTI_Exec );
    } else {
        return FTI_SCES;
    }

}

int FTI_UpdateHashBlocks(int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec) 
{
    FTI_ADDRVAL data_ptr = (FTI_ADDRVAL) FTI_Data[idx].ptr;
    FTI_ADDRVAL data_end = (FTI_ADDRVAL) FTI_Data[idx].ptr + (FTI_ADDRVAL) FTI_Data[idx].size;
    FTI_ADDRVAL data_size = (FTI_ADDRVAL) FTI_Data[idx].size;

    FTI_HashDiffInfo.dataDiff[idx].basePtr = data_ptr; 

    // if data size decreases
    if ( FTI_HashDiffInfo.dataDiff[idx].totalSize > data_size ) {
        
        long newNbBlocks = data_size/DIFF_BLOCK_SIZE;
        //dbg begin
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "id %d, newNbBlocks: %ld, nbBlocks: %ld\n", FTI_Data[idx].id, newNbBlocks,FTI_HashDiffInfo.dataDiff[idx].nbBlocks); 
        FTI_Print(str,FTI_INFO);
        // dbg end
        if ( data_size%DIFF_BLOCK_SIZE != 0 ) {
            newNbBlocks++;
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks = (FTIT_HashBlock*) realloc(FTI_HashDiffInfo.dataDiff[idx].hashBlocks, sizeof(FTIT_HashBlock)*(newNbBlocks));
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].hash = (unsigned char*) realloc( FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].hash, (MD5_DIGEST_LENGTH)*(newNbBlocks) );
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[newNbBlocks-1].hash = FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].hash + (newNbBlocks-1) * MD5_DIGEST_LENGTH;
            MD5_CTX ctx;
            MD5_Init(&ctx);
            MD5_Update(&ctx, (FTI_ADDRPTR) (data_ptr+(newNbBlocks-1)*DIFF_BLOCK_SIZE), data_size-(data_size/DIFF_BLOCK_SIZE * DIFF_BLOCK_SIZE));
            MD5_Final(FTI_HashDiffInfo.dataDiff[idx].hashBlocks[newNbBlocks-1].hash, &ctx);
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[newNbBlocks-1].dirty = false;
        }
        FTI_HashDiffInfo.dataDiff[idx].nbBlocks = newNbBlocks;
        FTI_HashDiffInfo.dataDiff[idx].totalSize = data_size;
    
    } else if ( FTI_HashDiffInfo.dataDiff[idx].totalSize < data_size ) {
        long newNbBlocks = data_size/DIFF_BLOCK_SIZE;
        // dbg begin
        char str[FTI_BUFS];
        snprintf(str, FTI_BUFS, "id %d, newNbBlocks: %ld, nbBlocks: %ld\n", FTI_Data[idx].id, newNbBlocks,FTI_HashDiffInfo.dataDiff[idx].nbBlocks); 
        FTI_Print(str,FTI_INFO);
        // dbg end
        if ( data_size%DIFF_BLOCK_SIZE != 0 ) {
            newNbBlocks++;
        }
        FTI_HashDiffInfo.dataDiff[idx].hashBlocks = (FTIT_HashBlock*) realloc(FTI_HashDiffInfo.dataDiff[idx].hashBlocks, sizeof(FTIT_HashBlock)*(newNbBlocks));    
        FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].hash = (unsigned char*) realloc( FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].hash, (MD5_DIGEST_LENGTH) * newNbBlocks );
        int hashIdx;
        long oldNbBlocks = FTI_HashDiffInfo.dataDiff[idx].nbBlocks;
        for(hashIdx = oldNbBlocks; hashIdx<newNbBlocks; ++hashIdx) {
            int hashBlockSize = ( (data_end - data_ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : data_end - data_ptr;
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].hash = FTI_HashDiffInfo.dataDiff[idx].hashBlocks[0].hash + hashIdx*MD5_DIGEST_LENGTH;
            MD5_CTX ctx;
            MD5_Init(&ctx);
            MD5_Update(&ctx, (FTI_ADDRPTR) data_ptr, hashBlockSize);
            MD5_Final(FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].hash, &ctx);
            FTI_HashDiffInfo.dataDiff[idx].hashBlocks[hashIdx].dirty = false; 
            data_ptr += DIFF_BLOCK_SIZE;
        }
        FTI_HashDiffInfo.dataDiff[idx].nbBlocks = newNbBlocks;
        FTI_HashDiffInfo.dataDiff[idx].totalSize = data_size;
    }
    return 0;
}

int FTI_GenerateHashBlocks( int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec ) {
   
    FTI_HashDiffInfo.dataDiff = (FTIT_DataDiffHash*) realloc( FTI_HashDiffInfo.dataDiff, (FTI_HashDiffInfo.nbProtVar+1) * sizeof(FTIT_DataDiffHash));
    assert( FTI_HashDiffInfo.dataDiff != NULL );
    FTI_ADDRVAL basePtr = (FTI_ADDRVAL) FTI_Data[idx].ptr;
    FTI_ADDRVAL ptr = (FTI_ADDRVAL) FTI_Data[idx].ptr;
    FTI_ADDRVAL end = (FTI_ADDRVAL) FTI_Data[idx].ptr + (FTI_ADDRVAL) FTI_Data[idx].size;
    long nbHashBlocks = ( ((FTI_Data[idx].size)%(DIFF_BLOCK_SIZE)) == 0 ) ? (FTI_Data[idx].size)/(DIFF_BLOCK_SIZE) : (FTI_Data[idx].size)/(DIFF_BLOCK_SIZE) + 1;
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].nbBlocks = nbHashBlocks;
    FTIT_HashBlock* hashBlocks = (FTIT_HashBlock*) malloc( sizeof(FTIT_HashBlock) * nbHashBlocks );
    assert( hashBlocks != NULL );
    // keep hashblocks array dense
    hashBlocks[0].hash = (unsigned char*) malloc( (MD5_DIGEST_LENGTH) * nbHashBlocks );
    assert( hashBlocks[0].hash != NULL );
    long cnt = 0;
    while( ptr < end ) {
        int hashBlockSize = ( (end - ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : end-ptr;
        hashBlocks[cnt].hash = hashBlocks[0].hash + cnt*MD5_DIGEST_LENGTH;
        MD5_CTX ctx;
        MD5_Init(&ctx);
        MD5_Update(&ctx, (FTI_ADDRPTR)ptr, hashBlockSize);
        MD5_Final(hashBlocks[cnt].hash, &ctx);
        hashBlocks[cnt].dirty = false;
        cnt++;
        ptr+=hashBlockSize;
    }
    assert( nbHashBlocks == cnt );
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].hashBlocks    = hashBlocks;
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].basePtr       = basePtr;
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].totalSize     = end - basePtr;
    FTI_HashDiffInfo.dataDiff[FTI_HashDiffInfo.nbProtVar].id            = FTI_Data[idx].id;
    FTI_HashDiffInfo.nbProtVar++;
    return FTI_SCES;
}

int FTI_ProtectPages ( int idx, FTIT_dataset* FTI_Data, FTIT_execution* FTI_Exec ) {
    char strdbg[FTI_BUFS];
    FTI_ADDRVAL first_page = FTI_GetFirstInclPage((FTI_ADDRVAL)FTI_Data[idx].ptr);
    FTI_ADDRVAL last_page = FTI_GetLastInclPage((FTI_ADDRVAL)FTI_Data[idx].ptr+FTI_Data[idx].size);
    FTI_ADDRVAL psize = 0;

    // check if dataset includes at least one full page.
    if (first_page < last_page) {
        psize = last_page - first_page + FTI_PageSize;
        if ( mprotect((FTI_ADDRPTR) first_page, psize, PROT_READ) == -1 ) {
            FTI_Print( "FTI was unable to register the pages", FTI_EROR );
            errno = 0;
            return FTI_NSCS;
        }
        // TODO no support for datasets that change size yet
        FTI_SignalDiffInfo.dataDiff = (FTIT_DataDiffSignal*) realloc( FTI_SignalDiffInfo.dataDiff, (FTI_SignalDiffInfo.nbProtVar+1) * sizeof(FTIT_DataDiffSignal));
        assert( FTI_SignalDiffInfo.dataDiff != NULL );
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].ranges = (FTIT_DataRange*) malloc( sizeof(FTIT_DataRange) );
        assert( FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].ranges != NULL );
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].rangeCnt       = 1;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].ranges->offset = (FTI_ADDRVAL) 0x0;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].ranges->size   = psize;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].basePtr        = first_page;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].totalSize      = psize;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].id             = FTI_Data[idx].id;
        FTI_SignalDiffInfo.nbProtVar++;

    // if not don't protect anything. NULL just for debug output.
    } else {
        FTI_SignalDiffInfo.dataDiff = (FTIT_DataDiffSignal*) realloc( FTI_SignalDiffInfo.dataDiff, (FTI_SignalDiffInfo.nbProtVar+1) * sizeof(FTIT_DataDiffSignal));
        assert( FTI_SignalDiffInfo.dataDiff != NULL );
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].rangeCnt       = 0;
        FTI_SignalDiffInfo.dataDiff[FTI_SignalDiffInfo.nbProtVar].id             = FTI_Data[idx].id;
        FTI_SignalDiffInfo.nbProtVar++;
        first_page = (FTI_ADDRVAL) NULL;
        last_page = (FTI_ADDRVAL) NULL;
    }

    // debug information
    snprintf(strdbg, FTI_BUFS, "FTI-DIFFCKPT: 'FTI_ProtectPages' - ID: %d, size: %lu, pages protect: %lu, addr: %p, first page: %p, last page: %p\n", 
            FTI_Data[idx].id, 
            FTI_Data[idx].size, 
            psize/FTI_PageSize,
            FTI_Data[idx].ptr,
            (FTI_ADDRPTR) first_page,
            (FTI_ADDRPTR) last_page);
    FTI_Print( strdbg, FTI_DBUG );
    return FTI_SCES;
}

FTI_ADDRVAL FTI_GetFirstInclPage(FTI_ADDRVAL addr) 
{
    FTI_ADDRVAL page; 
    page = (addr + FTI_PageSize - 1) & FTI_PageMask;
    return page;
}

FTI_ADDRVAL FTI_GetLastInclPage(FTI_ADDRVAL addr) 
{
    FTI_ADDRVAL page; 
    page = (addr - FTI_PageSize + 1) & FTI_PageMask;
    return page;
}

bool FTI_isValidRequest( FTI_ADDRVAL addr_val ) 
{
    
    if( addr_val == (FTI_ADDRVAL) NULL ) return false;

    FTI_ADDRVAL page = ((FTI_ADDRVAL) addr_val) & FTI_PageMask;

    if ( FTI_ProtectedPageIsValid( page ) && FTI_isProtectedPage( page ) ) {
        return true;
    }

    return false;

}

bool FTI_ProtectedPageIsValid( FTI_ADDRVAL page ) 
{
    // binary search for page
    bool isValid = false;
    int i;
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i){
        long LOW = 0;
        long UP = FTI_SignalDiffInfo.dataDiff[i].rangeCnt - 1;
        long MID = (LOW+UP)/2; 
        if ( FTI_RangeCmpPage(i, MID, page) == 0 ) {
            isValid = true;
        }
        while( LOW < UP ) {
            int cmp = FTI_RangeCmpPage(i, MID, page);
            // page is in first half
            if( cmp < 0 ) {
                UP = MID - 1;
            // page is in second half
            } else if ( cmp > 0 ) {
                LOW = MID + 1;
            }
            MID = (LOW+UP)/2;
            if ( FTI_RangeCmpPage(i, MID, page) == 0 ) {
                isValid = true;
                break;
            }
        }
        if ( isValid ) {
            break;
        }
    }
    return isValid;
}

bool FTI_isProtectedPage( FTI_ADDRVAL page ) 
{
    bool inRange = false;
    int i;
    for(i=0; i<FTI_SignalDiffInfo.nbProtVar; ++i) {
        if( (page >= FTI_SignalDiffInfo.dataDiff[i].basePtr) && (page <= FTI_SignalDiffInfo.dataDiff[i].basePtr + FTI_SignalDiffInfo.dataDiff[i].totalSize - FTI_PageSize) ) {
            inRange = true;
            break;
        }
    }
    return inRange;
}

int FTI_HashCmp( int varIdx, long hashIdx, FTI_ADDRPTR ptr, int hashBlockSize ) {
    if ( hashIdx < FTI_HashDiffInfo.dataDiff[varIdx].nbBlocks ) {
        unsigned char hashNow[MD5_DIGEST_LENGTH];
        MD5_CTX ctx;
        MD5_Init(&ctx);
        MD5_Update(&ctx, ptr, hashBlockSize);
        MD5_Final(hashNow, &ctx);
        if ( memcmp(hashNow, FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[hashIdx].hash, MD5_DIGEST_LENGTH) == 0 ) {
            return 0;
        } else {
            FTI_HashDiffInfo.dataDiff[varIdx].hashBlocks[hashIdx].dirty = true;
            return 1;
        }
    } else {
        return -1;
    }
}

int FTI_ReceiveDiffChunk(int id, FTI_ADDRVAL data_offset, FTI_ADDRVAL data_size, FTI_ADDRVAL* buffer_offset, FTI_ADDRVAL* buffer_size, FTIT_execution* FTI_Exec, FTIFF_dbvar* dbvar) {
   
    static bool init = true;
    static long pos;
    static FTI_ADDRVAL data_ptr;
    static FTI_ADDRVAL data_end;
    char strdbg[FTI_BUFS];
    if ( init ) {
        pos = 0;
        data_ptr = data_offset;
        data_end = data_offset + data_size;
        init = false;
    }
    
    int idx;
    long i;
    bool flag;
    // reset function and return not found
    if ( pos == -1 ) {
        init = true;
        return 0;
    }
   
    // if differential ckpt is disabled, return whole chunk and finalize call
    if ( !enableDiffCkpt ) {
        pos = -1;
        *buffer_offset = data_ptr;
        *buffer_size = data_size;
        return 1;
    }

    if ( diffMode == 0 ) {
        for(idx=0; (flag = FTI_HashDiffInfo.dataDiff[idx].id != id) && (idx < FTI_HashDiffInfo.nbProtVar); ++idx);
        if ( !flag ) {
            if ( !dbvar->hasCkpt ) {
                // set pos = -1 to ensure a return value of 0 and a function reset at next invokation
                pos = -1;
                *buffer_offset = data_ptr;
                *buffer_size = data_size;
                return 1;
            }
            long hashIdx = pos;
            int hashBlockSize = ( (data_end - data_ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : data_end - data_ptr;
            // advance *buffer_offset for clean regions
            while( FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == 0 ) {
                data_ptr += hashBlockSize;
                ++hashIdx;
                hashBlockSize = ( (data_end - data_ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : data_end - data_ptr;
            }
            /* if at call pointer to dirty region then data_ptr unchanged */
            *buffer_offset = data_ptr;
            *buffer_size = 0;
            // advance *buffer_size for dirty regions
            while( FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == 1 ) {
                *buffer_size += hashBlockSize;
                data_ptr += hashBlockSize;
                ++hashIdx;
                hashBlockSize = ( (data_end - data_ptr) > DIFF_BLOCK_SIZE ) ? DIFF_BLOCK_SIZE : data_end - data_ptr;
            }
            // check if we are at the end of the data region
            if ( FTI_HashCmp( idx, hashIdx, (FTI_ADDRPTR) data_ptr, hashBlockSize ) == -1 ) {
                //if ( data_ptr != data_end ) {
                //    FTI_Print("DIFF-CKPT: meta-data inconsistency: data size stored does not match runtime data size", FTI_WARN);
                //    init = true;
                //    return 0;
                //}
                if ( *buffer_size != 0 ) {
                    pos = -1;
                    return 1;
                } else {
                    init = true;
                    return 0;
                }
            }
            pos = hashIdx;
            return 1;
        }
    }

    if ( diffMode == 1 ) {
        for(idx=0; (flag = FTI_SignalDiffInfo.dataDiff[idx].id != id) && (idx < FTI_SignalDiffInfo.nbProtVar); ++idx);
        if( !flag ) {
            FTI_ADDRVAL base = FTI_SignalDiffInfo.dataDiff[idx].basePtr;
            // all memory dirty or not protected or first proper checkpoint
            if ( FTI_SignalDiffInfo.dataDiff[idx].rangeCnt == 0 || !FTI_Exec->hasCkpt ) {
                // set pos = -1 to ensure a return value of 0 and a function reset at next invokation
                pos = -1;
                *buffer_offset = data_ptr;
                *buffer_size = data_size;
                return 1;
            }
            for(i=pos; i<FTI_SignalDiffInfo.dataDiff[idx].rangeCnt; ++i) {
                FTI_ADDRVAL range_size = FTI_SignalDiffInfo.dataDiff[idx].ranges[i].size;
                FTI_ADDRVAL range_offset = FTI_SignalDiffInfo.dataDiff[idx].ranges[i].offset + base;
                FTI_ADDRVAL range_end = range_offset + range_size;
                FTI_ADDRVAL dirty_range_end; 
                // dirty pages at beginning of data buffer
                if ( data_ptr < range_offset ) {
                    snprintf(strdbg, FTI_BUFS, "FTI-DIFFCKPT: 'FTI_ReceiveDiffChunk:%d' - id: %d, idx-ranges: %lu, offset: %p, size: %lu\n",
                            __LINE__, 
                            FTI_SignalDiffInfo.dataDiff[idx].id, 
                            i,
                            range_offset,
                            range_size);
                    FTI_Print(strdbg, FTI_DBUG);
                    *buffer_offset = data_ptr;
                    *buffer_size = range_offset - data_ptr;
                    // at next call, data_ptr should be equal to range_offset of range[pos]
                    // and one of the next if clauses should be invoked
                    data_ptr = range_offset;
                    pos = i;
                    return 1;
                }
                // dirty pages after the beginning of data buffer
                if ( (data_ptr >= range_offset) && (range_end < data_end) ) {
                    if ( i < FTI_SignalDiffInfo.dataDiff[idx].rangeCnt-1 ) {
                        snprintf(strdbg, FTI_BUFS, "FTI-DIFFCKPT: 'FTI_ReceiveDiffChunk:%d' - id: %d, idx-ranges: %lu, offset: %p, size: %lu\n",
                                __LINE__, 
                                FTI_SignalDiffInfo.dataDiff[idx].id, 
                                i,
                                range_offset,
                                range_size);
                        FTI_Print(strdbg, FTI_DBUG);
                        data_ptr = FTI_SignalDiffInfo.dataDiff[idx].ranges[i+1].offset + base;
                        pos = i+1;
                        *buffer_offset = range_end;
                        *buffer_size = data_ptr - range_end;
                        return 1;
                        // this is the last clean range
                    } else if ( data_end != range_end ) {
                        snprintf(strdbg, FTI_BUFS, "FTI-DIFFCKPT: 'FTI_ReceiveDiffChunk:%d' - id: %d, idx-ranges: %lu, offset: %p, size: %lu\n",
                                __LINE__, 
                                FTI_SignalDiffInfo.dataDiff[idx].id, 
                                i,
                                range_offset,
                                range_size);
                        FTI_Print(strdbg, FTI_DBUG);
                        data_ptr = data_end;
                        pos = -1;
                        *buffer_offset = range_end;
                        *buffer_size = data_ptr - range_end;
                        return 1;
                    }
                }
                // data buffer ends inside clean range
                if ( (data_ptr >= range_offset) && (range_end >= data_end) ) {
                    break;
                }
            }
        }
    }
    // nothing to return -> function reset
    init = true;
    return 0;
}
