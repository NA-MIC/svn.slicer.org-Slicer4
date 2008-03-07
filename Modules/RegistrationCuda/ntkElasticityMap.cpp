#include "ntkElasticityMap.h"

ntkElasticityMap::ntkElasticityMap(ntkIntDimension mapSize){
  m_map=(float*)malloc(mapSize.x*mapSize.y*mapSize.z*sizeof(float));
  int i;
  for(i=0;i<mapSize.x*mapSize.y*mapSize.z;i++){
    *(m_map+i)=1.0;
  }
  m_mapSize=mapSize;
}

ntkElasticityMap::~ntkElasticityMap(){
  free(m_map);
}

float* ntkElasticityMap::getBuffer(){
  return m_map;
}

ntkIntDimension ntkElasticityMap::getMapSize(){
  return m_mapSize;
}

float ntkElasticityMap::getValue(int x, int y, int z){
  return *(m_map+z*m_mapSize.x*m_mapSize.y+y*m_mapSize.x+x);
}

void ntkElasticityMap::setValue(int x, int y, int z, float value){
  *(m_map+z*m_mapSize.x*m_mapSize.y+y*m_mapSize.x+x)=value;
}
