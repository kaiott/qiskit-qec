#ifndef __ArcTools__
#define __ArcTools__

#include<vector>
#include<tuple>
#include<map>
#include <set>

std::vector<int> check_nodes(
    std::vector<std::tuple<int, int, int, bool>> nodes, bool ignore_extra_logicals, bool minimal,
    std::map<std::tuple<int, int>, std::set<int>> cycle_dict,
    std::vector<std::tuple<int, int>> link_graph,
    std::map<int, std::vector<int>> link_neighbors,
    std::vector<int> z_logicals
    );

bool is_cluster_neutral(
    std::vector<std::tuple<int, int, int, bool>> nodes, bool ignore_extra_logicals, bool minimal,
    std::map<std::tuple<int, int>, std::set<int>> cycle_dict,
    std::vector<std::tuple<int, int>> link_graph,
    std::map<int, std::vector<int>> link_neighbors,
    std::vector<int> z_logicals,
    bool linear
    );

#endif