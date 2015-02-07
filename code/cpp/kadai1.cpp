/*
 * Copyright (c) 2015 Kentaro Wada
 * Released under the MIT license
 * https://github.com/YukinobuKurata/YouTubeMagicBuyButton/blob/master/MIT-LICENSE.txt
 *
 * author: www.kentaro.wada@gmail.com (Kentaro Wada)
 *
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <eigen3/Eigen/Core>
#include <boost/algorithm/string.hpp>
#include <boost/tuple/tuple.hpp>


Eigen::MatrixXd load_data(std::string filename);
void vector2matrix(std::vector< std::vector<double> > vec, Eigen::MatrixXd * mat);
int get_randint(int min, int max);


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


int get_randint(int min, int max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}


class LMS {
private:
    double eta;
    int iterations;
public:
    Eigen::VectorXd w;
    LMS() {};
    LMS(double _eta, int _iterations) {
        eta = _eta;
        iterations = _iterations;
    }
    void fit(Eigen::MatrixXd X, Eigen::VectorXd y) {
        Eigen::MatrixXd X_(X.rows(), X.cols()+1);
        X_ << X, Eigen::MatrixXd::Ones(X.rows(), 1);
        w = Eigen::VectorXd::Ones(X.cols()+1);

        for (int i=0; i<iterations; i++) {
            int choice = get_randint(0, X_.rows()-1);
            double predict = X_.row(choice) * w;
            double error = y[choice] - predict;
            Eigen::VectorXd dw = eta * error * X_.row(choice);
            w += dw;
        }
    }
    Eigen::VectorXd predict(Eigen::MatrixXd X) {
        int n_data = X.rows();
        Eigen::MatrixXd X_(X.rows(), X.cols()+1);
        X_ << X, Eigen::MatrixXd::Ones(n_data, 1);
        Eigen::VectorXd y_pred(X_.rows());
        for (int i=0; i<X_.rows(); i++) {
            double yp = X_.row(i) * w;
            std::vector<double> vec;
            vec.push_back((0-yp)*(0-yp));
            vec.push_back((1-yp)*(1-yp));
            std::vector<double>::iterator res = std::min_element(vec.begin(), vec.end());
            y_pred(i) = std::distance(vec.begin(), res);
        }
        return y_pred;
    }

    float score(Eigen::MatrixXd X, Eigen::VectorXd y_true) {
        Eigen::VectorXd y_pred = predict(X);
        int n_accurate=0;
        for (int i=0; i<y_pred.rows(); i++) {
            // count number of accurate prediction
            if (y_pred[i] == y_true[i]) n_accurate++;
        }
        float score = (float)n_accurate / (float)y_pred.rows();
        return score;
    }
};


boost::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>
get_kadai1_dataset()
{
    // train data
    Eigen::MatrixXd X_train1 = load_data("../../data/Train1.txt");
    Eigen::MatrixXd X_train2 = load_data("../../data/Train2.txt");
    Eigen::VectorXd y_train1 = Eigen::VectorXd::Zero(X_train1.rows());
    Eigen::VectorXd y_train2 = Eigen::VectorXd::Ones(X_train2.rows());
    Eigen::MatrixXd X_train(X_train1.rows()+X_train2.rows(),
                            X_train1.cols());
    X_train << X_train1,
               X_train2;
    Eigen::VectorXd y_train(y_train1.rows()+y_train2.rows());
    y_train << y_train1,
               y_train2;
    // test data
    Eigen::MatrixXd X_test1 = load_data("../../data/Test1.txt");
    Eigen::MatrixXd X_test2 = load_data("../../data/Test2.txt");
    Eigen::VectorXd y_test1 = Eigen::VectorXd::Zero(X_test1.rows());
    Eigen::VectorXd y_test2 = Eigen::VectorXd::Ones(X_test2.rows());
    Eigen::MatrixXd X_test(X_test1.rows()+X_test2.rows(),
                           X_test1.cols());
    X_test << X_test1,
              X_test2;
    Eigen::VectorXd y_test(y_test1.rows()+y_test2.rows());
    y_test << y_test1,
              y_test2;
    return boost::make_tuple(X_train, y_train, X_test, y_test);
}


int main(int argc, char* argv[])
{
    boost::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd> dataset = get_kadai1_dataset();
    Eigen::MatrixXd X_train = dataset.get<0>();
    Eigen::VectorXd y_train = dataset.get<1>();
    Eigen::MatrixXd X_test = dataset.get<2>();
    Eigen::VectorXd y_test = dataset.get<3>();

    // fit and predict
    LMS lms = LMS(0.001, 10000);
    lms.fit(X_train, y_train);
    // Eigen::VectorXd y_pred = lms.predict(X_test);
    double score = lms.score(X_test, y_test);
    printf("%lf,", score);

    double a = 1. / lms.w[1] * (0.5 - lms.w[2]);
    double b = 1. / lms.w[1] * (-lms.w[0]);
    printf("y=%lfx+%lf\n", a, b);
    return 0;
}

