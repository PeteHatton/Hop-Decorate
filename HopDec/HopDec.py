import os
import copy
import pickle as pkl
import numpy as np

from .Input import getParams
from .State import *
from .Model import *
from .Utilities import log
from . import Redecorate
from . import NEB
from . import MD


def mainCMD(comm):

    from mpi4py import MPI
    MPI.pickle.THRESHOLD = 0

    rank = comm.Get_rank()
    size = comm.Get_size()

    # each rank gets its own single-rank communicator for LAMMPS
    localComm = comm.Split(color=rank, key=rank)

    comm.barrier()

    params = getParams()

    model = None
    if rank == 0:

        filename = 'model-checkpoint_latest.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model = pkl.load(f)
        else:
            model = setupModel(params, comm=localComm)

    k = 0
    redecorate = params.redecorateTransitions
    search = params.modelSearch

    while redecorate or search:

        if redecorate:

            transitions = []
            if rank == 0:
                transitions = model.transitionList
            transitions = comm.bcast(transitions, root=0)

            for t, trans in enumerate(transitions):
                if not trans.redecoration:

                    filename = Redecorate.main(trans, params, comm=comm)

                    if rank == 0:
                        model.transitionList[t].redecoration = filename
                        checkpoint(model)
                        checkpoint(model, filename=f'model-checkpoint_{len(model)}.pkl')

            redecorate = 0

        comm.barrier()

        if search:

            if rank == 0:
                workDistribution = model.workDistribution(size)
                initialState = workDistribution[0]
                for r in range(1, size):
                    comm.send(workDistribution[r], dest=r, tag=0)
            else:
                initialState = comm.recv(source=0, tag=0)

            initialState = copy.deepcopy(initialState)
            _, finalState, flag = MD.main(initialState,
                                          params,
                                          comm=localComm,
                                          maxMDTime=params.segmentLength,
                                          rank=rank,
                                          verbose=True)

            connection = None
            if flag:
                log('NEB', f'rank: {rank} Running NEB', 0)
                connection = NEB.main(initialState, finalState, params, comm=localComm)

            connections = comm.gather(connection, root=0)

            if rank == 0:

                status = model.update(workDistribution, connections=connections)

                if status:
                    checkpoint(model)
                    checkpoint(model, filename=f'model-checkpoint_{len(model)}.pkl')
                    if params.redecorateTransitions:
                        redecorate = 1

                k += 1
                if k % params.checkpointInterval == 0:
                    checkpoint(model)

                minTime = np.min([state.time for state in model.stateList])
                if minTime > np.inf:  # params.minTimeCutoff
                    search = 0

            search = comm.bcast(search, root=0)
            redecorate = comm.bcast(redecorate, root=0)

        comm.barrier()

    if rank == 0:
        checkpoint(model)
        checkpoint(model, filename=f'model-checkpoint_{len(model)}.pkl')
        log('Model', 'All work finished')
