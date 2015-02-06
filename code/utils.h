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
