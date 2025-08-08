

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// TASK to i
#define TASK_TO_I(x) (((size_t)(-1 + sqrt(1 + (x - 1) * 8))) / 2)
// TASK to j
#define TASK_TO_J(x, i) ((x) - ((i + 1) * (i)) / 2 - 1)
// i,j to TASK
#define IJ_TO_TASK(i, j) ((((i) + 1) * (i)) / 2 + 1 + (j))

void readDegDistFile(const std::string &filename, std::vector<double> &degrees, std::vector<double> &counts)
{
    std::ifstream infile(filename);
    if (!infile)
    {
        std::cerr << "Failed to open file.\n";
        return;
    }
    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        int key, value;
        if (!(iss >> key >> value))
        {
            continue; // skip malformed lines
        }

        degrees.push_back(static_cast<double>(key));
        counts.push_back(static_cast<double>(value));
    }
    infile.close();
}

void readLocationFile(const std::string &filename, std::vector<double> &counts)
{
    std::ifstream infile(filename);
    if (!infile)
    {
        std::cerr << "Failed to open file.\n";
        return;
    }
    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        int value;
        if (!(iss >> value))
        {
            continue; // skip malformed lines
        }
        counts.push_back(static_cast<double>(value));
    }
    infile.close();
}

double getEdgeCost(size_t locId, const std::vector<double> &degrees, const std::vector<double> &counts, double total)
{
    size_t i = locId + 1;

    size_t ix = TASK_TO_I(i);
    size_t jx = TASK_TO_J(i, ix);

    double c = NORMAL_WORKLOAD;

    int key_i = degrees[ix];
    int value_i = counts[ix];

    int key_j = degrees[jx];
    int value_j = counts[jx];

    if (ix == jx)
    {
        c = value_i * (value_i + 1) / 2 * ((key_i * key_j) / total);
    }
    else
    {
        c = value_i * value_j * ((key_i * key_j) / total);
    }

    return c;
}

double getLocationCost(size_t locId, const std::vector<double> &counts)
{
    double c = 1 + counts[locId] * (counts[locId] - 1) / 2; // Example cost function, can be modified as needed
    return c;
}

int main(int argc, char *argv[])
{
    int N = 100000000; // Default value for N
    int P = 8;
    if (argc > 1)
    {
        N = std::stoi(argv[1]);
    }
    if (argc > 3)
    {
        P = std::stoi(argv[3]);
    }
    std::cout << "Using " << P << " threads.\n";

    std::vector<double> degrees;
    std::vector<double> counts;
    double total = 0;

    if (argc > 2)
    {
#if 0
        readDegDistFile(argv[2], degrees, counts);
        N = degrees.size() * (degrees.size() + 1) / 2;
        for (int i = 0; i < degrees.size(); ++i)
        {
            total += degrees[i] * counts[i];
        }
#else
        readLocationFile(argv[2], counts);
        N = counts.size();

        std::cout << "N = " << N << std::endl;
#endif
    }

    // Example: print the loaded degree distribution
    std::cout << "Total degree: " << total << "\n";
    std::cout << "Total edges : " << total / 2 << "\n";

    std::cout << "Degree distribution size: " << degrees.size() << "\n";

    size_t TASK = degrees.size() * (degrees.size() + 1) / 2;

    size_t task = 0;
    double cost = 0;

    size_t TASK_PER_THREAD = ceil((TASK + 1) / (double)P);

    // Fill in Costs using normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    double mean = 1000.0;  // example mean
    double stddev = 100.0; // example standard deviation
    std::normal_distribution<> d(mean, stddev);

    for (long i = 1; i <= TASK; ++i)
    {
#if 0
        gen.seed(i - 1);
        double c = d(gen);
        cost += c;
#else

#if 0
        double c = getEdgeCost(locId, degrees, counts, total);
#else
        double c = getLocationCost(i - 1, counts);
#endif

        cost += c;
#endif
        if ((i + 1) % TASK_PER_THREAD == 0)
        {
            std::cout << "Task " << i << " Rank " << (i) / TASK_PER_THREAD << "\t" << "Cost: " << cost << " completed.\n";
            cost = 0;
        }
    }
    std::cout << "Rank " << (P - 1) << "\t" << "Cost: " << cost << " completed.\n";
    std::cout << "Expected cost : " << cost << "\n";

    std::cout << "Total tasks.  : " << task << "\n";
    std::cout << "Expected tasks: " << TASK << "\n";

    return 0;
}
