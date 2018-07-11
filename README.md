One-sided Queue in MPI
=====================

After learning about passive target synchronisation I figured I'd try to find a way to remove the 'master' part from master-slave MPI queue implementations by storing the next starting point on one rank and updating it using atomic RMA operations (MPI_Fetch_and_op). This should be simpler to implement than a master-slave style setup, and should avoid the wasted resources that come with an MPI rank doing nothing apart from waiting for other ranks to talk to it.

This should prove useful for applications that have embarassingly parallel sections that need to be combined in some way afterwards. Should also prove more performant than supplying each MPI rank a fixed range to work over in the case of imbalanced workloads, assuming the cost of updating/synchronising the memory remotely is not too large.