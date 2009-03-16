#pragma once

#include <stdio.h>
#include <stdlib.h>
//#include <strings.h>
#include <string.h>
#include <math.h>


#include "vtkImageData.h"


#define MAX_COLS 512
#define MAX_DEPTH 201
#define MAX_STRING_SIZE 100
#define MAXX 1024
#define MAXY 1024
#define MAXZ 160
#undef TRUE
#undef FALSE

typedef enum {FALSE, TRUE} BOOLEAN;

/*
 * This is the include file to define the spherical pixel structure used for
 *    cell edge detection.
 *
 * Dean P McCullough
 * Oct 4, 04
 *
 * RIMAGEDEF which holds the rectangular image
 *
 * IMAGEDEF which holds the spherical image:
 *
 * rows, represents Pi/2 to -Pi/2
 * cols, represent 0 to 2Pi
 * rad, represents radius
 * conn, represents where a pixel has for the next row 
 *    2 followers = 1
 *    1 follower  = 0
 *    .5 followers = -1
 * prad, the radius to the pole position
 *
 * SURFACE which will hold the resulting "best" surface 
 *
 * 
 */

typedef struct RIMAGEDEF
{
  char NAME[40];
  int nx;
  int ny;
  int nz;
  double aspratio;
  int cent[3];
  int bound[3];
  unsigned char data[MAXX][MAXY][MAXZ];
}  RIMAGEDEF;



typedef struct IMAGEDEF
{
  char NAME[40];
  int nrows;
  int ncols;
  int nrad;
  int reduce;
  double minrat;
  double maxrat;
  int conn[1+MAX_COLS/2];
  int data[1+MAX_COLS/2][MAX_COLS][MAX_DEPTH];
  char prad;
}  IMAGEDEF;

typedef struct SURFACE
{
  char NAME[40];
  int nrows;
  int ncols;
  int nrad;
  int conn[1+MAX_COLS/2];
  int surface[1+MAX_COLS/2][MAX_COLS]; 
} SURFACE;

class vtkCellWallVisSeg
{
public:
        vtkCellWallVisSeg();
        ~vtkCellWallVisSeg();
        
//member functions
public:
        void readImage(char const* immage);
        bool afterLoadingInit();
        void setCellCenter(double xyzv[4]);
        void setCellEdge(double xyzv[4]);
        void addCellEdge(double xyzv[4]);
        void addCellEdge3D(int x, int y);
        void ResetSegmentationResult(vtkImageData* imagePtr, int CellID);
        
        // pass a vtkImage and a label number, the voxels in the vtk image will be set to match the label
        // for any voxels which had been set in the segmentation
        void RenderSegmentationResult(vtkImageData* imagePtr, int CellID);
        
        bool resamp();
        
        RIMAGEDEF& getRimage(){return rimage;};
        unsigned char* getPixbuf(){return pixbuf;};
        
        void compute2DBoundary(int CellID);
        void compute3DSeg(int CellID);
        
private:
        int interp(double pt[3]);
        void flagpt(int i, int j, int k);
        void flagpt6(int iii, int jjj);
        
        int qdynam();
        int dynam1();

        int qdynam0(int, int);
        int qdynam2(int, int);
        int ddynam0(int, int);
        int ddynam1(int, int);
        int ddynam2(int, int);
        int ddynam3(int, int);
        int ddynam4(int, int);

        // added by jusub
        int qdynams2(int,int);
        int ddynams2(int,int);
        int ddynams3(int,int);
        int ddynams4(int,int);
        
        void dispedge(int id);
        void dispresult(int numb, int CellID);
        
private:
        //global variables from the original code
        RIMAGEDEF rimage;
        IMAGEDEF image;
        IMAGEDEF image2;
        SURFACE result;
        int zflg; // do not avoid zero value in segmentation
        int dflg; // do not compute difference image
        int funct;
        int loaded;
        int edits;
        unsigned char *pixbuf, *pixbuf2;
        int level;
        int elev;
        int maxlevel;
        int levelsize;
        int imwidth;
        int imheight;
        int pwidth;
        int slab;
        int wlab;
        int aaaa;
        FILE *logfile;
        int score;
};

