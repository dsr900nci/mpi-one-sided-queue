
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#define QUEUE_LIMIT 14000
// Phase 1: Fixed increment known to all at runtime
#define QUEUE_INCREMENT 10
#define WORK_TIME 0
#define WORK_UNCERTAINTY 1000000


int main(int argc, char **argv)
{


/*   We have some fixed sized queue, and the completion time of each packet of work
   associated with a queue slot is uniformly distributed around some mean. Therefore,
   assigning each MPI rank a fixed range to work on will lead to significant load
   imbalance. We wish to ameliorate this by having each rank request new work whenever
   it has completed the work assigned. Usually, this is implemented by having a 'master'
   rank listening for completion and assinging each 'slave' rank a new range on request.
   Here, we're exposing a piece of memory on one rank, and using Advanced Passive Target 
   Synchronisation to have each rank (including the one with the memory exposed) retrieve
   its new starting point. At first, the range will be fixed and known to all ranks at
   the start of the job. Then, each rank will calculate the range it is to work on using
   some function known to all ranks at run time, and update the starting point accordingly. */


/* Notes so far:
   
   The startup seems to be a bit unstable, perhaps we need to distribute the starting values
   to each rank another way, then use the one-sided queue after the first bunch of work is
   complete. - An MPI_Barrier call fixed this issue, it was due to overlap in the MPI_Win_lock and 
   MPI_Win_lock_all calls.

   MPI_Win_lock_all is slow for processes on the root node with OpenMPI >=2.1. Seems to
   improve slightly with OMPI_MCA_mpi_leave_pinned=1
   Seems fine in OpenMPI 2.1.2, though MPI_Win_lock_all does run slower in Score-P
   Segfaults in OpenMPI 2.0.4
   MPI_Win_flush_all is slow on all ranks using OpenMPI 1.10.7

   MPI_Win_flush_all operations are extremely slow in Intel-MPI on nodes other than 
   the root node.   

   Further reading shows that using Win_*_all is overkill, since we only need to access
   memory on one rank. We can just use Win_lock/flush/unlock, instead, which hopefully helps
   performance a bit.
   Results:
   OpenMPI 3.1.0 : MPI_Win_lock time increases to 35 seconds.
   OpenMPI 3.0.1 : Same timing as MPI_Win_lock_all
   OpenMPI 2.1.3 : Same timing as MPI_Win_lock_all
   OpenMPI 2.0.4 : Still segfaults
   Intel MPI: All ranks except first process on each node hangs
   

*/

  // MPI stuff
  int mpi_rank, mpi_size;
  int ierr;

  // Window stuff
  int win_size = 0;
  int disp_unit = sizeof(int);
  int *win_mem;
  MPI_Win win;
  
  // Queue stuff
  int this_start = 0;
  int this_end = 0;
  int queue_increment = QUEUE_INCREMENT;
  int work_time;
  int i;
  time_t t;
  
  MPI_Init(&argc, &argv);

  
  MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
  MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );

  // Seed the rando number generator
  srand( ( mpi_rank + 2 ) * (unsigned) time(&t) );


  //   Set up the shared memory
  if ( mpi_rank == 0 ) win_size = sizeof(int) ;

  ierr = MPI_Win_allocate( win_size, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &win_mem, &win );

  // Set up a small fixed window to start with
  this_start = mpi_rank * QUEUE_INCREMENT;
  this_end   = this_start + QUEUE_INCREMENT;
  // Start after the first bunch of work is done
  if ( mpi_rank == 0 ) {
    ierr = MPI_Win_lock( MPI_LOCK_EXCLUSIVE, 0, 0, win );
    // memset( win_mem, (mpi_size+1)*QUEUE_INCREMENT, sizeof(int) );
    *win_mem = (mpi_size+1)*QUEUE_INCREMENT;
    ierr = MPI_Win_unlock( 0, win );
  }

  // Lock all to start
  ierr = MPI_Barrier( MPI_COMM_WORLD );
  //  ierr = MPI_Win_lock_all( 0, win );
  ierr = MPI_Win_lock( MPI_LOCK_SHARED, 0, 0, win );
  
  while ( this_start < QUEUE_LIMIT ) {
    
    printf( "Rank %d has %d to %d\n", mpi_rank, this_start, this_end );
    // Do the 'work'
    for( i = this_start; i < this_end; ++i ) {
      work_time = WORK_TIME + rand()%WORK_UNCERTAINTY;
      //      if ( i == this_start ) printf("Rank: %d, work time:%d\n", mpi_rank, work_time );
      usleep( work_time ); 

    }

    // Grab the next starting point
    if ( mpi_rank == 0 ) {
      this_start = *win_mem;
      *win_mem += queue_increment;
    }
    else {
      ierr = MPI_Fetch_and_op( &queue_increment, &this_start, MPI_INT, 0, 0, MPI_SUM, win );
    }

    // Update the window for everyone
    ierr = MPI_Win_flush( 0, win );
    //ierr = MPI_Win_sync( win );
    this_end = this_start + queue_increment;      
  }

  ierr = MPI_Win_unlock( 0, win );
  ierr = MPI_Win_free( &win );


  // Finish up
  ierr = MPI_Finalize();

}
    
  
