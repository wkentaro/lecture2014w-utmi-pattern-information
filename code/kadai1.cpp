#include <iostream>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <boost/algorithm/string.hpp>

std::vector< std::vector<double> >
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

    infile.close();

    return data;
}

Eigen::MatrixXd
vector2matrix(std::vector< std::vector<double> > vec)
{
    Eigen::MatrixXd mat(vec.size(), vec[0].size());
    for (int i=0; i<vec.size(); i++) {
        for (int j=0; j<vec[i].size(); j++) {
            mat.block(i, j, 1, 1) << vec[i][j];
        }
    }
    return mat;
}

template<typename T>
void display_data(T data)
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
    Eigen::MatrixXd mat;
    mat = vector2matrix(data);
    std::cout << mat;
}
