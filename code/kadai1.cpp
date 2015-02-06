#include <iostream>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <boost/algorithm/string.hpp>
#include "utils.h"

Eigen::MatrixXd load_data(std::string filename);
void vector2matrix(std::vector< std::vector<double> > vec, Eigen::MatrixXd * mat);


Eigen::MatrixXd
load_data(std::string filename)
{
    // open file stream
    std::ifstream infile(filename);
    if (!infile) std::cout << "unable to load file" << std::endl;

    // get data from file
    std::string str;
    std::vector< std::vector<double> > data;
    while (getline(infile, str)) {
        // split data string and convert to double
        std::string::size_type sz;
        double val1 = stod(str, &sz);
        double val2 = stod(str.substr(sz));
        // store data
        std::vector<double> row;
        row.push_back(val1);
        row.push_back(val2);
        data.push_back(row);
    }

    // close file stream
    infile.close();

    // convert vector to matrix
    Eigen::MatrixXd mat(data.size(), data[0].size());
    vector2matrix(data, &mat);

    return mat;
}


void vector2matrix(
    std::vector< std::vector<double> > vec,
    Eigen::MatrixXd * mat
    )
{
    for (int i=0; i<vec.size(); i++) {
        for (int j=0; j<vec[i].size(); j++) {
            mat->block(i, j, 1, 1) << vec[i][j];
        }
    }
}


int main(int argc, char* argv[]) {
    Eigen::MatrixXd mat;
    mat = load_data("../data/Train1.txt");
    std::cout << mat;
}

