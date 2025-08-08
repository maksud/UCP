#include "csv.h"
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
#include <utility>
#include <vector>

#define TIMESTEPS 60

#define DEBUG 0

typedef unsigned int AgentType;
typedef unsigned int LocationType;
typedef unsigned int CountType;

typedef struct
{
    AgentType agent_id;
    uint64_t presence;
} AgentPresence;

typedef struct Contact
{
    AgentType target;
    AgentType source;
    int duration;

    bool operator<(const Contact &a) const
    {
        return target == a.target ? source < a.source : target < a.target;
    }
} Contact;

typedef struct
{
    std::string agent;
    int begin_m;
    int end_m;
    std::string location;
} Activity;

int popcount64c(uint64_t x)
{
    x -= (x >> 1) & 0x5555555555555555;                             // put count of each 2 bits into those 2 bits
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333); // put count of each 4 bits into those 4 bits
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f;                        // put count of each 8 bits into those 8 bits
    return (x * 0x0101010101010101) >> 56;                          // returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
}

class ActivityTable
{
  public:
    ActivityTable()
    {
        nAgents_ = 0;
        nLocations_ = 0;
        nActivities_ = 0;

        // Count the number of AgentPresence by Hour
        for (int hour = 0; hour < 24; hour++)
        {
            countAgentPresenceByHour_[hour] = 0;
        }
    }

    void insertActivity(Activity log)
    {
        nActivities_++;
        auto agent_it = agentToIdMap_.find(log.agent);
        if (agent_it == agentToIdMap_.end())
        {
            agentToIdMap_[log.agent] = nAgents_;
            idToAgentMap_[nAgents_] = log.agent;
            nAgents_++;
        }
        AgentType agent_id = agentToIdMap_[log.agent];

        auto loc_it = locationToIdMap_.find(log.location);
        if (loc_it == locationToIdMap_.end())
        {
            locationToIdMap_[log.location] = nLocations_;
            idToLocationMap_[nLocations_] = log.location;
            nLocations_++;
        }
        LocationType location_id = locationToIdMap_[log.location];

        int start_h = log.begin_m / 60;
        int end_h = log.end_m / 60;

        for (int h = start_h; h <= end_h; ++h)
        {
            // Create empty presence vector for an agent.
            std::bitset<TIMESTEPS> ag_pr_vec;

            // Fetch the activity log.
            int hour_start = 60 * h;
            int hour_end = 60 * (h + 1);

            int begin_min, end_min;

            begin_min = log.begin_m <= hour_start ? 0 : log.begin_m - hour_start;
            end_min = log.end_m >= hour_end ? 59 : log.end_m - hour_start;

            for (int i = begin_min; i <= end_min; ++i)
            {
                ag_pr_vec.set(i);
            }

            AgentPresence ag_presence = {agent_id, ag_pr_vec.to_ullong()};
            hourLocationToAgentMap_[h][location_id].push_back(ag_presence);
            countAgentPresenceByHour_[h]++;
        }
    }

    AgentType getNumberOfAgents() const
    {
        return nAgents_;
    }

    LocationType getNumberOfLocations() const
    {
        return nLocations_;
    }

    void saveAgentMap(std::ofstream &fout)
    {
        fout << "agent_name,agent_row_id\n";
        for (const auto &x : agentToIdMap_)
        {
            fout << x.first << "," << x.second << std::endl;
        }
    }

    void generateColocationCPU(int hour)
    {
        contacts_.clear();

        char filename[32];
        std::snprintf(filename, sizeof(filename), "colocated_agents_sizes_%02d.txt", hour);
        std::ofstream hour_out(filename);

        for (LocationType locId = 0; locId < nLocations_; locId++)
        {
            auto outerIt = hourLocationToAgentMap_.find(hour);
            if (outerIt != hourLocationToAgentMap_.end())
            {
                auto innerIt = outerIt->second.find(locId);
                if (innerIt != outerIt->second.end())
                {
                    std::vector<AgentPresence> &colocatedAgents = innerIt->second;
                    // Save the size of colocatedAgents to a file

                    if (hour_out.is_open())
                    {
                        hour_out << colocatedAgents.size() << std::endl;
                    }
                }
            }
        }
        hour_out.close();

        printf("Total Contacts: %lu\n", contacts_.size());
        // std::sort(contacts_.begin(), contacts_.end());
    }

    void generateColocation(std::string &dir_name)
    {
        for (int hour = 0; hour < 24; hour++)
        {
            auto t3 = std::chrono::high_resolution_clock::now();

            std::cout << "------------------------------------------------------------------------------------------" << std::endl;

            generateColocationCPU(hour);

            auto t4 = std::chrono::high_resolution_clock::now();
            auto dur_hourly = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
            std::cout << "Hour " << hour << "-> Time taken: " << dur_hourly.count() << std::endl;

            std::cout << "------------------------------------------------------------------------------------------" << std::endl;

#if 0
            auto t5 = std::chrono::high_resolution_clock::now();
            if (std::filesystem::exists(dir_name))
            {
                std::ofstream mat_out;
                std::string output_file = dir_name + "/Hour_" + std::to_string(hour) + ".txt";
                mat_out.open(output_file);

                mat_out << "row,column,value\n";
                for (int i = 0; i < contacts_.size(); i++)
                {

                    Contact &c = contacts_[i];
                    mat_out << c.target << "," << c.source << "," << c.duration << "\n";
                }
                mat_out.close();
            }
            else
            {
                std::cout << "Failed to save edgelists.\n";
            }
            auto t6 = std::chrono::high_resolution_clock::now();
            auto dur_save = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5);
            std::cout << "Time to save edgelist for Hour " << hour << ": " << dur_save.count() << std::endl;
#endif
        }
    }

    int getTableSize() const
    {
        return nActivities_;
    }

  private:
    std::map<int, std::map<LocationType, std::vector<AgentPresence>>> hourLocationToAgentMap_;
    std::map<int, CountType> countAgentPresenceByHour_;

    // Output
    std::vector<Contact> contacts_;

    // Some helper Data Structures
    std::map<std::string, AgentType> agentToIdMap_;
    std::map<std::string, LocationType> locationToIdMap_;

    std::map<AgentType, std::string> idToAgentMap_;
    std::map<LocationType, std::string> idToLocationMap_;

    AgentType nAgents_;
    LocationType nLocations_;
    CountType nActivities_;

    // types and constants used in the functions below
    // uint64_t is an unsigned 64-bit integer variable type (defined in C99 version of C language)
    const uint64_t m1 = 0x5555555555555555;  // binary: 0101...
    const uint64_t m2 = 0x3333333333333333;  // binary: 00110011..
    const uint64_t m4 = 0x0f0f0f0f0f0f0f0f;  // binary:  4 zeros,  4 ones ...
    const uint64_t m8 = 0x00ff00ff00ff00ff;  // binary:  8 zeros,  8 ones ...
    const uint64_t m16 = 0x0000ffff0000ffff; // binary: 16 zeros, 16 ones ...
    const uint64_t m32 = 0x00000000ffffffff; // binary: 32 zeros, 32 ones
    const uint64_t h01 = 0x0101010101010101; // the sum of 256 to the power of 0,1,2,3...
};

ActivityTable readCSV(const std::string &input_filename, int clock_offset = 0)
{
    ActivityTable table = ActivityTable();
    io::CSVReader<4> infile(input_filename);
    infile.read_header(io::ignore_extra_column, "agent", "begin_m", "end_m", "act_location");
    std::string agent, location;
    float begin_t, end_t;
    while (infile.read_row(agent, begin_t, end_t, location))
    {
        int begin_m = std::trunc(begin_t), end_m = std::trunc(end_t);
        begin_m -= clock_offset;
        end_m -= clock_offset;
        end_m -= end_m % 1440 == 0 ? 1 : 0; // So that end_m does not equals to multiples of 24*60 = 1440

        if (begin_m >= 0 && end_m < 1440)
        {
            Activity record = {agent, begin_m, end_m, location};
            table.insertActivity(record);
        }
        else
        {
            // std::cout << "Check Entry: " << agent << "\t" << begin_t << "\t" << end_t << "\t" << location <<
            // std::endl;
        }
    }
    return table;
}

int main(int argc, char **argv)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0)
    {
        std::cout << "This code computes the hourly collocation matrices for a 24-hour activity log.\nIt takes 2 "
                     "arguments:\n1. Input filepath\n2. Clock offset (i.e. what minute does the day start?)\nUsage "
                     "example: ./executable ../activity_log_wednesday.csv 4320\n";
        return 0;
    }
    assert((argc >= 3) && "Accepts two arguments. Try option --help.");

    // Input parameters
    std::string input_filename(argv[1]);
    int clock_offset = std::atoi(argv[2]);

    bool save_to_file = false;

    std::cout << "Filename: " << input_filename << std::endl;

    // Read CSV
    ActivityTable act_table = readCSV(input_filename, clock_offset);

    int n_agents = act_table.getNumberOfAgents();
    int n_locations = act_table.getNumberOfLocations();

    std::cout << "Number of agents: " << n_agents << "\n"
              << "Number of locations: " << n_locations << "\n"
              << "Number of activity records: " << act_table.getTableSize() << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dur_read = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << "Time taken to read in activity logs: " << dur_read.count() << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();

    std::string dir_name;
    if (save_to_file)
    {

        // Create directory for saving edgelists.
        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        // dir_name = "../hourly_contact_edgelists_" + std::to_string(ts);
        dir_name = "./hourly_contact_edgelists_ma";

        std::cout << "Time text: " << dir_name << std::endl;
        std::filesystem::create_directory(dir_name);

        // Save agent id map
        if (std::filesystem::exists(dir_name))
        {
            std::ofstream ag_map_out;
            std::string ag_outfile = dir_name + "/agent_id_map.txt";
            ag_map_out.open(ag_outfile);
            if (ag_map_out.is_open())
            {
                act_table.saveAgentMap(ag_map_out);
            }
            else
            {
                std::cout << "Failed to create file for agent id map.\n";
            }
        }
        else
        {
            std::cout << "Failed to create directory for saving edgelists.\n";
        }
    }

    std::cout << "Total Initialization: " << dur_read.count() << std::endl;

    for (int i = 0; i < 1; i++)
    {
        auto t8 = std::chrono::high_resolution_clock::now();

        act_table.generateColocation(dir_name);

        auto t7 = std::chrono::high_resolution_clock::now();
        auto dur_comp = std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t8);
        std::cout << std::endl << "Tick " << i << " Total computation time for 24 hours: " << dur_comp.count() << std::endl << std::endl;
    }
    return 0;
}
