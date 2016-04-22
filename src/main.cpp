#include <iostream>

namespace l2n
{

int main(int argc, char** argv)
{
    std::cout << "Hello World !\n";
    return 0;
}

}

int main(int argc, char** argv)
{
    return l2n::main(argc, argv);
}