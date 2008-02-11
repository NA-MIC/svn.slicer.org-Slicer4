#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>

int main(int argc, char** argv)
{

  if (argc < 6) 
    return 0;
  
  std::string fileName = argv[1];
  std::string format = argv[2];
  
  std::stringstream s;

  unsigned int resolution[3];

  s << argv[3];
  s >> resolution[0];
  s << argv[4];
  s >> resolution[1];
  s << argv[5];
  s >> resolution[2];
  
  std::cout << "InputFileName: " << fileName <<
    "Format: " << format <<
    "Resolution: " << resolution[0] << "x" << resolution[1] << "x" << resolution[2] << std::endl;

  

}

