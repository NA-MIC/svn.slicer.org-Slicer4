#ifndef NTKELASTICITYMAP_H
#define NTKELASTICITYMAP_H

#include <stdio.h>
#include "ntkProperties.h"

class ntkElasticityMap{
 public:
  ntkElasticityMap(ntkIntDimension mapSize);
  ~ntkElasticityMap();
  float* getBuffer();
  ntkIntDimension getMapSize();
  float getValue(int x, int y, int z);
  void setValue(int x, int y, int z, float value);
 protected:
  float* m_map;
  ntkIntDimension m_mapSize;
};

#endif
