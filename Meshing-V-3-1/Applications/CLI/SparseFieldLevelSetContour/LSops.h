/*=========================================================================

  Program:   SparseFieldLevelSetContour
  Module:    $HeadURL$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Brigham and Women's Hospital (BWH) All Rights Reserved.

  See License.txt or http://www.slicer.org/copyright/copyright.txt for details.

==========================================================================*/
#ifndef LSOPS
#define LSOPS

#include "MeshOps.h"
#include "MeanCurvatureEnergy.h"

#include <list>

class SparseFieldLS
{
public:
  SparseFieldLS( MeshData* data, const std::list<int>& C, MeshEnergy* mergy )
{
L_z = C;
meshdata = data;
energy = mergy;
phi = vector<double>(data->MeanCurv.size());
CleanLZ();
InitSphere();
dDirection = -1.0;
}
SparseFieldLS( MeshData* data, const std::list<int>& C, 
const std::list<int>& Lp1, const std::list<int>& Ln1, const std::list<int>& Lp2,
const std::list<int>& Ln2, const std::vector<int>& map )
{
L_z = C;
meshdata = data;
L_z = C; L_n1 = Ln1; L_p1 = Lp1; L_n2 = Ln2; L_p2 = Lp2;
int numVerts = map.size();
point_type = vector<int>(numVerts);
phi        = vector<double>(numVerts);
for( unsigned int i = 0; i < map.size(); i++ ) {
  point_type[i] = map[i];
  phi[i]        = double( map[i] );
}
energy = new MeanCurvatureEnergy( data );
dDirection = 1.0;
}
std::vector<int> Evolve( int iterations = 1 );


private:
// the different layers for sparse-field method
MeshEnergy* energy;
MeshData* meshdata;
std::list<int> L_z;
std::list<int> L_n1;
std::list<int> L_p1;
std::list<int> L_n2;
std::list<int> L_p2;
double dDirection;

vector<int> point_type;
vector<double> phi;
void SelfUnion( vector<int>& vec );
void SelfUnion( list<int>& vec );
void DropIdx( const vector<int>& L_zp, const vector<int>& L_zn, vector<int>& L_z );
void DropIdx( const list<int>& L_zp, const list<int>& L_zn, list<int>& L_z );
void AppendIdx( const vector<int>& src, vector<int>& dst );

public:
void SwitchDirection()    { dDirection *= -1.0; }
std::vector<double>* GetPhi()  { return &phi; };
std::list<int>*  GetPtrLZ()  { return &L_z; };

vector<int> EvolveRecalc( );
// "sphere": assume that the mesh is homeomorphic to a sphere
bool InitSphere(); // determine what the other layers are on the mesh given L_z
bool InitFast(); // TODO: a fast way of doing this (local search...)
void CleanLZ(); // delete extraneous LZ points
};

#endif
