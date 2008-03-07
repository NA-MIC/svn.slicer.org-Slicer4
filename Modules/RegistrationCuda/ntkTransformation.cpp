#include "ntkTransformation.h"

ntkTransformation::ntkTransformation(){

}

void ntkTransformation::printMatrix(){
  int i,j;
  printf("\n");
  for(i=0;i<4;i++){
    printf("[ ");
    for(j=0;j<4;j++){
      printf("%f ",m_tMatrix->GetElement(i,j));
    }
    printf(" ]\n");
  }
  printf("\n");
}
