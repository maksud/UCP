#include "mpi.h"
#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

#define TIMESTEPS 60

#define DEBUG 0

typedef unsigned int AgentType;
typedef unsigned int LocationType;
typedef unsigned int CountType;

typedef int TaskType;

typedef int ProcessorType;
typedef std::map<int, LocationType> MapBoundary;

class BoundaryMessage
{

  public:
    int id;
    double from;
    double to;
    double capacity;

    BoundaryMessage(int start = -1, double startFrom = -1, double startTo = -1, double endFrom = -1)
    {
        this->id = start;
        this->from = startFrom;
        this->to = startTo;
        this->capacity = endFrom;
    }

    void fill(int id = -1, double from = -1, double to = -1, double capacity = -1)
    {
        this->id = id;
        this->from = from;
        this->to = to;
        this->capacity = capacity;
    }

    friend std::ostream &operator<<(std::ostream &out, const BoundaryMessage &x)
    {
        out << "<" << x.id << ", " << x.from << " -- " << x.to << ">\t [" << x.capacity << "]\t";
        return (out);
    }

    BoundaryMessage clone()
    {
        return (BoundaryMessage(this->id, this->from, this->to, this->capacity));
    }

    void clone(BoundaryMessage *data)
    {
        this->id = data->id;
        this->from = data->from;
        this->to = data->to;
        this->capacity = data->capacity;
    }
};

// Debug Framework
int NORMAL_WORKLOAD = 100;
int IMBALANCED_WORKLOAD = 100000;
int idx_imbalanced = 0;

class UCP
{
  public:
    UCP(LocationType nLocations)
    {
        nTasks_ = nLocations;
    }

    void mpiInit()
    {
        // Get the number of processes
        MPI_Comm_size(MPI_COMM_WORLD, &P_);
        // Get the rank of the process
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

        // MPI Data Type
        const int nitems = 4;
        int blocklengths[4] = {1, 1, 1, 1};
        MPI_Datatype types[6] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
        MPI_Aint offsets[4];

        offsets[0] = offsetof(BoundaryMessage, id);
        offsets[1] = offsetof(BoundaryMessage, from);
        offsets[2] = offsetof(BoundaryMessage, to);
        offsets[3] = offsetof(BoundaryMessage, capacity);

        MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpiTypeBoundaryMessage_);
        MPI_Type_commit(&mpiTypeBoundaryMessage_);

        TaskType N_TASKS_PER_RANK = (TaskType)ceil(((double)nTasks_) / P_);
        lowerBound_ = rank_ * N_TASKS_PER_RANK;
        upperBound_ = std::min((rank_ + 1) * N_TASKS_PER_RANK, nTasks_);

        // Hack that works
        if (upperBound_ == nTasks_)
        {
            upperBound_ = nTasks_ + 1;
        }

        printf("Rank %d - lowerBound_ = %d, upperBound_= %d\n", rank_, lowerBound_, upperBound_);

        size_ = upperBound_ - lowerBound_;
        C_ = new double[size_];
    }

    void balanceConsecutiveLoads()
    {
        double zi = C_[size_ - 1];

        /// STEP (4): Exclusize Prefix Sum of zi -> Zi
        MPI_Exscan(&zi, &Zi_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /// STEP (6): Sum of Zi_ -> Z
        MPI_Allreduce(&zi, &Z_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        double Z_bar = Z_ / (double)P_;

        makePartition(Z_bar, nTasks_);
    }

    void freeMemory()
    {
        delete[] C_;
    }

    // MPI Specific
    ProcessorType rank_;
    ProcessorType P_;

    // Load Balancing
    TaskType nTasks_;

    TaskType lowerBound_;
    TaskType upperBound_;
    TaskType size_;
    BoundaryMessage startBoundary_;
    BoundaryMessage endBoundary_;
    double *C_;

  private:
    void findBoundary(int start, int end, double capacity)
    {
        if (start > end)
            return;

        /// Distance of middle index from lower bound (or upper bound)
        int b = (end - start + 1) / 2;
        int m = start + b;

        int pMm1 = (int)((C_[m - 1] + Zi_ - 0.00001) / capacity);
        int pM = (int)((C_[m] + Zi_ - 0.00001) / capacity);

        if (pMm1 != pM)
        {
            BoundaryMessage message;
            if (pMm1 < P_ - 1)
            {
                // Send a Message to Compute the initial part of Task: m+lowerBound_ to Processor pMm1
                message.fill(m + lowerBound_, 0, ((pMm1 + 1) * capacity - C_[m - 1] - Zi_), C_[m] - C_[m - 1]);
                MPI_Bsend(&message, 1, mpiTypeBoundaryMessage_, pMm1, 1, MPI_COMM_WORLD);
            }

            for (int k = pMm1 + 1; k < pM; k++)
            {
                message.fill(m + lowerBound_, (k * capacity - C_[m - 1] - Zi_), ((k + 1) * capacity - C_[m - 1] - Zi_), C_[m] - C_[m - 1]);
                // Send A PJ Message to k
                // Send C_ Null NJ Message to k
                MPI_Bsend(&message, 1, mpiTypeBoundaryMessage_, k, 1, MPI_COMM_WORLD);
                message.fill();
                MPI_Bsend(&message, 1, mpiTypeBoundaryMessage_, k, 2, MPI_COMM_WORLD);
            }
            if (pM < P_)
            {
                // Send a Message to Compute the last part of Task: m+lowerBound_ to Processor pMm1
                message.fill(m + lowerBound_, (pM * capacity - (C_[m - 1] + Zi_)), (C_[m] - C_[m - 1]), C_[m] - C_[m - 1]);
                MPI_Bsend(&message, 1, mpiTypeBoundaryMessage_, pM, 2, MPI_COMM_WORLD);
            }
        }

        int pS = (int)((C_[start - 1] + Zi_) / capacity);
        int pE = (int)((C_[end] + Zi_) / capacity);

        if (pS != pMm1)
            findBoundary(start, m - 1, capacity);
        if (pE != pM)
            findBoundary(m + 1, end, capacity);
    }

    void makePartition(double Z_bar, LocationType nLocations)
    {
        /// STEP (7): Find n_k
        /// If the boundary is between two processors keep the higher one
        if ((ProcessorType)(Zi_ / Z_bar) != (ProcessorType)((C_[0] + Zi_) / Z_bar))
        {
            dictKn_[(ProcessorType)((C_[0] + Zi_) / Z_bar)] = lowerBound_;
        }
        findBoundary(0, size_ - 1, Z_bar);

        MPI_Status status;
        if (rank_ < P_ - 1)
            MPI_Recv(&endBoundary_, 1, mpiTypeBoundaryMessage_, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        else
        {
            endBoundary_.fill(nLocations, -1, -1);
        }

        if (rank_ > 0)
            MPI_Recv(&startBoundary_, 1, mpiTypeBoundaryMessage_, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
        else
        {
            startBoundary_.fill(0, 0, C_[0], C_[0]);
        }

        if (rank_ == 0)
        {
            if (startBoundary_.id == endBoundary_.id)
                startBoundary_.id = -1;
        }
    }

    MPI_Datatype mpiTypeBoundaryMessage_;
    MPI_Status status_;
    MapBoundary dictKn_;

    double Zi_;
    double Z_;
};

int main(int argc, char **argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the processor name
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print a message from each process
    std::cout << "Hello from processor " << processor_name << ", rank " << world_rank << " out of " << world_size << " processors." << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    int TASK_NUM = 100000000;

    UCP ucp(TASK_NUM);
    ucp.mpiInit();

    double *C = ucp.C_;

    // Fill in Costs
    for (TaskType locId = ucp.lowerBound_; locId < ucp.upperBound_; locId++)
    {
        double c = NORMAL_WORKLOAD;

        if (locId == idx_imbalanced)
            c = IMBALANCED_WORKLOAD;

        if (locId == ucp.nTasks_)
        {
            c = 0;
        }

        if (locId == ucp.lowerBound_)
        {
            C[locId - ucp.lowerBound_] = c;
        }
        else
        {
            C[locId - ucp.lowerBound_] = c + C[locId - ucp.lowerBound_ - 1];
        }
    }

    ucp.balanceConsecutiveLoads();
    ucp.freeMemory();

    size_t totalWorkLoad = 0;

    // PART 1:
    if (ucp.rank_ >= 0)
    {
        int x = ucp.startBoundary_.id;

        if (x >= 0 && x < ucp.nTasks_)
        {

            double eStart = ucp.startBoundary_.from;
            double eEnd = ucp.startBoundary_.to;

            totalWorkLoad += eEnd - eStart;

            std::cout << "#1. Rank " << ucp.rank_ << " Task " << x << " SubTask: eStart " << eStart << " eEnd " << eEnd << std::endl;
        }
    }

    // Part 2:
    if (ucp.startBoundary_.id >= 0)
    {
        int ll = ucp.startBoundary_.id + 1;
        int hl = ucp.endBoundary_.id - 1;

        for (int i = ll; i <= hl; i++)
        {
            int c = NORMAL_WORKLOAD;
            if (i == idx_imbalanced)
                c = IMBALANCED_WORKLOAD;
            totalWorkLoad += c;
        }

        std::cout << "#2. Rank " << ucp.rank_ << " Task " << ll << " to " << hl << std::endl;
    }

    // Part 3:
    // Process End Portions
    if (ucp.rank_ <= ucp.P_ - 1)
    {
        int x = ucp.endBoundary_.id;

        if (x >= 0 && x < ucp.nTasks_)
        {

            double eStart = ucp.endBoundary_.from;
            double eEnd = ucp.endBoundary_.to;

            totalWorkLoad += eEnd - eStart;

            std::cout << "#3. Rank " << ucp.rank_ << " Task " << x << " SubTask: eStart " << eStart << " eEnd " << eEnd << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "FINAL: Rank " << ucp.rank_ << " Work " << totalWorkLoad << std::endl;

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
