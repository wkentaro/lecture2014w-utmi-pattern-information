#include <iostream>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <boost/algorithm/string.hpp>

std::vector< std::vector<double> >
load_data(std::string filename)
{
    std::ifstream infile(filename);
    if (!infile) std::cout << "unable to load file" << std::endl;

    std::string str;
    std::vector< std::vector<double> > data;
    while (getline(infile, str)) {
        std::string::size_type sz;
        double val1 = stod(str, &sz);
        double val2 = stod(str.substr(sz));
        std::vector<double> row;
        row.push_back(val1);
        row.push_back(val2);
        data.push_back(row);
    }
    return data;
}

void display_data(std::vector< std::vector<double> > data)
{
    for (int i=0; i<data.size(); i++) {
        for (int j=0; j<data[i].size(); j++) {
            std::cout << data[i][j] << ", ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::vector< std::vector<double> > data;
    data = load_data("../data/Train1.txt");
    display_data(data);
}