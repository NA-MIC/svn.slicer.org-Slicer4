#include "vtkCellWallVisSeg.h"
#include <string>
#include <iostream>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

vtkCellWallVisSeg::vtkCellWallVisSeg()
{
        zflg=0; // do not avoid zero value in segmentation
        dflg=0; // do not compute difference image
        funct = 0;
        loaded = 0;
        edits = 0;
        pixbuf=NULL;
        pixbuf2=NULL;
        slab = 0;
        wlab = 1;
        aaaa = 0;
        score = 0;
}

vtkCellWallVisSeg::~vtkCellWallVisSeg()
{
}

void vtkCellWallVisSeg::readImage(char const* immage)
{
//  double noize(double sigma); commented out by yanling
  FILE *ics_file, *ids_file;

  char filename[81];
  char ia[60], ib[50], ic[50];
  char *zline;
  int c;
  int ii, i1, i2, i3, i4;
  int i,j,k;
  double x, yy, zz;
  int imax, imin;
  int hist[256];

  strcpy(filename,immage);
  //strcat(filename,".ics");
  printf("immage='%s' filename='%s'\n",immage,filename);
  if((ics_file=fopen(filename,"r")) ==  NULL)
    {
       strcpy(filename,immage);
       strcat(filename,".ICS");
       if((ics_file=fopen(filename,"r")) ==  NULL)
         {
           printf("*** file %s could not be opened. ***\n",filename);
           exit(0);
         }
    }

  zline=(char *)malloc(MAX_STRING_SIZE*sizeof(char));
  if( zline == NULL )
    {
      printf("malloc failed for line\n");
      exit(1);
    }
  memset((char *)zline,'\0',MAX_STRING_SIZE*sizeof(char));

  ii = 0;
  while(fgets( (char *)zline,MAX_STRING_SIZE,ics_file) != NULL)
    {
      ii++;
      sscanf(zline,"%s %s",ia,ib);
      if(strncmp(ia,"filename",8)==0) 
        {
          strncpy(rimage.NAME, ib,39); 
          printf("File name = %s  \n",ib);
          strcat(ib,".log");
//        logfile = open_file(ib,"a");
          logfile = fopen(ib,"a");
          
          if (logfile != NULL)//added by yanling
          {
                printf("Log file in %s\n",ib);

                fprintf(logfile,"\n\nCell_wall, V25\nStarting session\nZero Overlap = %d\n",zflg);
                //dat(logfile); //commented out by jusub 
                fflush(logfile);
          }
        }
      if(strncmp(ia,"layout",6) == 0)
        {
          if(strncmp(ib,"sizes",5) == 0)
            {
              sscanf(zline,"%s %s %d %d %d %d",ia,ib,&i1, &i2, &i3, &i4);
              printf("Immage dimension = %d %d %d\n",i2,i3,i4);
              if(i1 != 8)
                {
                  printf("Code does not yet handle other than 8 bits/voxel\n");
                  exit(1);
                }
              rimage.nx = i2;
              rimage.ny = i3;
              rimage.nz = i4;
            } 
        }
      if(strncmp(ia,"representation",14) == 0)
        {
          if(strncmp(ib,"byte_order",10) == 0)
            {
              sscanf(zline,"%s %s %d",ia,ib,&i1);
              if(i1 != 1)
                {
                  printf("Code does not yet handle other than byte order 1\n");
                  exit(1);
                }
            }
          if(strncmp(ib,"compression",11) == 0)
            {
              sscanf(zline,"%s %s %s",ia,ib,ic);
              if(strncmp(ic,"uncompressed",12) != 0)
                {
                  printf("Code does not yet handle other than uncompressed\n");
                  exit(1);
                }
            }
        }
     
    // added by jusub ====================================================
    if(strncmp(ia,"history",7) == 0)
        {
          if(strncmp(ib,"aspectRatio",11) == 0)
            {
              sscanf(zline,"%s %s %lf",ia,ib,&rimage.aspratio);
              printf("Aspect Ratio = %lf \n",rimage.aspratio);
            } 
        }
    if(strncmp(ia,"history",7) == 0)
        {
          if(strncmp(ib,"dyedType",8) == 0)
            {
                  sscanf(zline,"%s %s %s",ia,ib,ic);
                  if(strncmp(ic,"surface",7)==0){
                          printf("dyed type = surface \n");
                  dflg = 0;
                  }
                  else  
                        {
                        printf("dyed type = volume \n");
                        dflg=1;
                        }
                }           
        }
     // end of addition ===================================================

      memset((char *)zline,'\0',MAX_STRING_SIZE*sizeof(char));
    }

  fclose (ics_file);

  fprintf(logfile,"Size = %d %d %d\n",rimage.nx,
          rimage.ny, rimage.nz);
  fflush(logfile);


  if(i2 > MAXX)
    {
      printf("ERROR:  Insufficient space to hold x axis. %d > %d\n",
             i2, MAXX);
      exit(2);
    }

  if(i3 > MAXY)
    {
      printf("ERROR:  Insufficient space to hold y axis. %d > %d\n",
             i3, MAXY);
      exit(2);
    }

  if(i4 > MAXZ)
    {
      printf("ERROR:  Insufficient space to hold z axis. %d > %d\n",
             i4, MAXZ);
      exit(2);
    }

  fprintf(logfile,"Delta flg = %d\n",dflg);
  fflush(logfile);

  // set a few other constants for converting to spherical coordinates 
  image.nrows = 128;
  image.ncols = 256;

  // Will try to set this automatically in resamp image.nrad = 101;  
  image.minrat = .3;
  image.maxrat = 4.0;
 
  

  // trim off the .ics extension to pass only the pathname and proper filename
    int filenamelength;
    filenamelength = strlen(immage);
    strncpy(filename,immage,filenamelength-4);
    filename[filenamelength-4]='\0';
    // add the .ids extension to the binary component of the datafile
    strcat(filename,".ids");
    //printf("immage='%s' filename='%s'\n",immage,filename);
  
  if((ids_file=fopen(filename,"r")) ==  NULL)
    {
      // trim of the .ics extension and add .IDS in case the extenion has 
      // capital letters in the filename
      strncpy(filename,immage,filenamelength-4);
      filename[filenamelength-4]='\0';
         strcat(filename,".IDS");
       if((ids_file=fopen(filename,"r")) ==  NULL)
         {
           //printf("immage='%s' filename='%s'\n",immage,filename);
           printf("*** file %s could not be opened. ***\n",filename);
           exit(0);
         }
    }

/* commented out by jusub
 * no intensity scaling, do not avoid zero value in segmentation by default
 * spatial aspect ratio and whether it will compute difference image is 
 * now specified in ics file

  printf("\nAspect Ratio of Z axis to XY plain:  ");
  scanf("%s", ia);
  if (strlen(ia) <1)
    {
      fprintf(stderr,"Error: No aspect ratio supplied; 1.0 will be assumed\n");
      rimage.aspratio = 1.0;
    }
  else
    {
      x = atof(ia);
      rimage.aspratio = x;
      printf("Aspect ratio = %f\n",x);
      fprintf(logfile,"Aspect ratio = %f\n",x);

    }

  printf("\nDo a first difference of the image: (y,n)  ");
  dflg = 0;
  scanf("%s",ia);
  if(strlen(ia) < 1)
    printf("No reply received, will assume No\n");
  if(tolower(ia[0]) == 'y')
    {
      dflg = 1;
      printf("Will use a first difference of the image.\n");
    }

*/
  zz = 0.0;

  for(i=0;i<256;i++)
    hist[i] = 0;

  imax = 0;
  imin = 512;

  if(zz != 0.0)
    for(k=0;k<i4;k++)
      for(j=0;j<i3;j++)
        for(i=0;i<i2;i++)
          {
            c =  fgetc(ids_file);
//          c = c + noize(zz); commented out by jusub
            if(c < 0) c = 0;
            if(c > 255) c = 255;
            hist[c]++;
            if(c < imin) imin = c;
            if(c > imax) imax = c; 
            rimage.data[i][j][k] = c;
          }
  else
  
    for(k=0;k<i4;k++)
      for(j=0;j<i3;j++)
        for(i=0;i<i2;i++)
          {
            c = fgetc(ids_file);
            hist[c]++;
            if(c < imin) imin = c;
            if(c > imax) imax = c;
            rimage.data[i][j][k] = c;
          }



  fclose (ids_file);

  printf("Max = %d  Min = %d \n",imax,imin);

 /* commented out by jusub

        printf("\n Ratio for scaling: ");
  scanf("%s",ia);
  if(strlen(ia) < 1)
    yy = 0.0;
  else
    yy = atof(ia);

  if(yy >+ 1.0)
    {
      printf("ERROR: ratio for scaling %f is greater than 1.  Will be set to .02\n",yy);
      yy = .02;
    }
  if(yy < 0.0)
    {
     printf("ERROR: ratio for scaling %f is less than 0.  Will be set to .02\n",yy);
      yy = .02;
    } 

  j = yy * i2 * i3 * i4;
  i = 0;
  imin = 0;
  printf(" %f  per cent = %d \n",100*yy,j);
  for(imin = 0;i<=j;imin++)
    i=i+hist[imin];
  imin--;

  imax = 255;
  i=0;
  for(imax = 255;i<=j;imax--)
    i = i+hist[imax];
  imax++;

  x = 253.0/(imax-imin);
  printf("ScaMax = %d  ScaMin = %d  Scale = %f\n",imax,imin,x);
  fprintf(logfile,"%.2f percent  ScaMax = %d  ScaMin = %d\n",100*yy,imax,imin);

  // Set so that minimum is at least 1, saving 0 for a "do not use"
  //   flag, unless we want 0 to flag "do not use".
  

  i1 = 0;
  printf("\nAvoid voxels with 0 value (Y,N):   ");
  scanf("%s",ia);
  if(strlen(ia) < 1)
    printf("No reply received, will assume No\n");
  if(tolower(ia[0]) == 'y')
    {
      i1 = 1;
      printf("Will avoid voxels with 0 value.\n");
      zflg = 1;
      fprintf(logfile,"Avoid voxels with 0 value.\n");
      fflush(logfile);

    }

  imin = imin-1;
  
  for(k=0;k<i4;k++)
    for(j=0;j<i3;j++)
      for(i=0;i<i2;i++)
        {
          if((rimage.data[i][j][k] != 0) || (i1 == 0))
            {
              ii = x*(rimage.data[i][j][k]-imin);
              if(ii < 1) ii = 1;
              if(ii > 253) ii = 253;
              rimage.data[i][j][k] = ii;
            }
        }

  fprintf(logfile,"Scale = %f\n",x);
  fflush(logfile);

  printf("Finished reading image\n");
  
*/
}

bool vtkCellWallVisSeg::afterLoadingInit()
{
        //initialize variables
        imheight = rimage.ny;
    imwidth = rimage.nx;
    maxlevel = rimage.nz;
    level = rimage.nz/2;
        levelsize= imwidth*imheight*3; //added by jusub for pixbuf
    elev = 0;
    pwidth = (int)(2.0 * image.maxrat * MAX_DEPTH / (image.maxrat - image.minrat));
         
        if(pixbuf!=NULL) delete pixbuf; 
        pixbuf = (unsigned char*)malloc(sizeof(unsigned char)*rimage.ny*rimage.nx*3*maxlevel);   
    if(pixbuf == NULL) 
    {
       printf("Error in malloc for image space\n");
       return false;
    }
         
        if(pixbuf2!=NULL) delete pixbuf2; 
    pixbuf2 = (unsigned char*)malloc(sizeof(unsigned char)*pwidth*pwidth*3);     
    if(pixbuf2 == NULL) 
    {
       printf("Error in malloc for Poler image space\n");
       return false;
    }
         
        return true;
}

void vtkCellWallVisSeg::setCellCenter(double xyzv[4])
{
  // conversion for legacy c code  
  // legacy c code use (0,0) at left top corner.
  int i,j,offset;
  i=(int)xyzv[0];
  j=(int)(rimage.ny-xyzv[1]-1);
  level=(int)xyzv[2];
          
  //from legacy c code in even_cb() ======
  rimage.cent[0] = i;
  rimage.cent[1] = j;
  rimage.cent[2] = level;
  // ========================================
}

void vtkCellWallVisSeg::setCellEdge(double xyzv[4])
{
        // set one edge point of cell boundary and compute 2D boundary    

        // conversion for legacy c code   
        int i,j,offset;
        i=(int)xyzv[0];
        j=(int)(rimage.ny-xyzv[1]-1);
        level=(int)xyzv[2];
  
        // from legacy c code in even_cb() ======
        rimage.bound[0] = i;
        rimage.bound[1] = j;
        rimage.bound[2] = level;
        resamp();
        loaded = 1;
}

bool vtkCellWallVisSeg::resamp()
{
  int i,j,jj,k;
  int nrow, knum;
  int cent[3], bound[3];
  double rad[MAX_DEPTH];
  int ddd[MAX_DEPTH + 4],xtmp;
  double mm,dm, dis;
  double basevec[3], baservec[3], basecvec[3], uvec[3];
  double pt[3], aspratio;
//  int interp();
  double pi=M_PI;
  double x,y;
  int ncol, nrad;
  double minrat, maxrat;

  ncol = image.ncols;
  /*  nrad = image.nrad; */
  minrat = image.minrat;
  maxrat = image.maxrat;

  /* Check some parameters */

  if(ncol > MAX_COLS)
    {
      printf("ERROR in resamp: Number of cols greater than max allowed %d %d\n",
             ncol, MAX_COLS);
      return false;
    }

  /*  if(nrad > MAX_DEPTH)
    {
      printf("ERROR in resamp: Number of radius greater than max allowed %d %d\n",
             nrad, MAX_DEPTH);
      exit(1);
    }
  */

  /*  Verify that ncol is a power of 2 */
  nrow = ncol/2;

  printf("In resamp nrow = %d %d ncol = %d\n",nrow,image.nrows,ncol);

  i=16;
  for(j=0;j<10;j++)
    {
      if(i == nrow) goto match;
      i = i<<1;
    }
  fprintf(stderr," *** ERROR *** in resamp.  nrow not a power of 2\n");
  fprintf(stderr," nrow = %d  Max 8192 Min 16\n",nrow);
  return false;

 match:;

  for(i=0;i<nrow;i++)
    image.conn[i] = 0;

  image.conn[0] = 1;
  image.conn[1] = 1;
  image.conn[nrow-1] = -1;
  image.conn[nrow-2] = -1;

  k = ncol;
  while(k > 16)
    {
      j = k/12 + 1;
      if(j==3) j=4;
      image.conn[j] = 1;
      image.conn[nrow-1-j] = -1;
      k=k/2;
    }

  j=ncol/12 + 1;
  if(j==3) j=4;
  image.reduce = nrow-(j+2);

  strcpy(image.NAME, rimage.NAME); 

  cent[0] = rimage.cent[0];
  cent[1] = rimage.cent[1];
  cent[2] = rimage.cent[2];
  bound[0] = rimage.bound[0];
  bound[1] = rimage.bound[1];
  bound[2] = rimage.bound[2];
  aspratio = rimage.aspratio;

  printf("Cent = %d %d %d   Edge = %d %d %d\n",cent[0], cent[1],cent[2],bound[0],bound[1],bound[2]);

  if(cent[2] != bound[2])
    {
      printf("ERROR in resamp: Boundry point and center not on same z plane. %d %d\n",
             cent[2],bound[2]);
      //return;// added by jusub
          return false;//added by yanling
      // exit(1);
    }

  dis = (bound[0]-cent[0])*(bound[0]-cent[0]) + 
    (bound[1]-cent[1])*(bound[1]-cent[1]);
  dis = sqrt(dis);

  dm = maxrat - minrat;
  mm = 1.0 - minrat;

  nrad = 1.0*dm*dis;
  if(nrad > MAX_DEPTH) nrad = MAX_DEPTH;
  image.nrad = nrad;
  
  printf("Ncol = %d, Nrad = %d\n",ncol,nrad);

  i = (mm*nrad / dm) +.5;
  image.prad = i;

  dm = dm/nrad;
  
  for(i=0; i<nrad;i++)
    rad[i] = dis*(minrat + i*dm);

  printf("%f %f %f %d %d\n",dis,rad[0],rad[nrad-1],image.prad,nrad);

  for(i=0;i<3;i++)
    {
      basevec[i] = (bound[i] - cent[i])/dis;
      uvec[i] = basevec[i];
    }
 
  for(i=0; i<nrad; i++)
    {
      for(jj=0; jj<2; jj++)
        pt[jj] = cent[jj] + rad[i]*uvec[jj];
      pt[2] = cent[2] + rad[i]*uvec[2]/aspratio;

      ddd[i+2] = interp(pt);
    }

  ddd[0] = ddd[1] = ddd[nrad+2] = ddd[nrad+3] = 0;

  if(dflg == 0)
    for(i=0;i<nrad; i++)
      image.data[0][0][i] = ddd[i+2];
  else      
    for(i=0; i<nrad; i++)
      {
        if(ddd[i+2] < 1)
          xtmp = ddd[i+2];
        else if(ddd[i+3] < 1)
          xtmp = 64 + ddd[i] + 2*ddd[i+1] - (3*ddd[i+2]);
        else if(ddd[i+4] < 1)
          xtmp = 64 + ddd[i] + 2*ddd[i+1] - (3*ddd[i+3]);
        else
          xtmp = 64 + ddd[i] + 2*ddd[i+1] - (ddd[i+4] + 2*ddd[i+3]);
        if(xtmp < 1) xtmp = 1;
        if(xtmp > 253) xtmp = 253;
        image.data[0][0][i] = xtmp;
      }

  if(nrad < MAX_DEPTH)
    for(i=nrad; i<MAX_DEPTH; i++)
      image.data[0][0][i] = 0;

  uvec[0] = -basevec[0];
  uvec[1] = -basevec[1];
  uvec[2] = basevec[2];

  for(i=0; i<nrad; i++)
    {
      for(jj=0; jj<2; jj++)
        pt[jj] = cent[jj] + rad[i]*uvec[jj];
      pt[2] = cent[2] + rad[i]*uvec[2]/aspratio;

      ddd[i+2] = interp(pt);
    }

  ddd[0] = ddd[1] = ddd[nrad+2] = ddd[nrad+3] = 0;

  if(dflg == 0)
    for(i=0;i<nrad; i++)
      image.data[nrow][0][i] = ddd[i+2];
  else

    for(i=0; i<nrad; i++)
      {
        if(ddd[i+2] < 1)
          xtmp = ddd[i+2];
        else if(ddd[i+3] < 1)
          xtmp = 64 + ddd[i] + 2*ddd[i+1] - (3*ddd[i+2]);
        else if(ddd[i+4] < 1)
          xtmp = 64 + ddd[i] + 2*ddd[i+1] - (3*ddd[i+3]);
        else
          xtmp = 64 + ddd[i] + 2*ddd[i+1] - (ddd[i+4] + 2*ddd[i+3]);
        if(xtmp < 1) xtmp = 1;

        if(xtmp > 253) xtmp = 253;

        image.data[nrow][0][i] = xtmp;
      }

  if(nrad < MAX_DEPTH)
    for(i=nrad; i<MAX_DEPTH; i++)
      image.data[nrow][0][i] = 0;

  /* for each row */
  knum = 8;
  for(j=1;j<nrow;j++)
    {

      /* generate base unit vector for this row */
      x = j*pi/nrow;

      baservec[0] = basevec[0]*cos(x);
      baservec[1] = basevec[1]*cos(x);
      baservec[2] = 0;
      basecvec[0] = -basevec[1]*sin(x);
      basecvec[1] =  basevec[0]*sin(x);
      basecvec[2] = 0;
      dis = basecvec[0]*basecvec[0] + basecvec[1]*basecvec[1];
      dis = sqrt(dis);

      /* for each col in this row */
      for(k=0;k<knum;k++)
        {

          /* generate unit vector for this scan line */
          y = (k*pi*2)/knum;
          uvec[0] = baservec[0] + basecvec[0] * cos(y);
          uvec[1] = baservec[1] + basecvec[1] * cos(y);
          uvec[2] = dis*sin(y);

          for(i=0; i<nrad; i++)
            {
              for(jj=0; jj<2; jj++)
                pt[jj] = cent[jj] + rad[i]*uvec[jj];
              pt[2] = cent[2] + rad[i]*uvec[2]/aspratio;

              ddd[i+2] = interp(pt);
            }

          ddd[0] = ddd[1] = ddd[nrad+2] = ddd[nrad+3] = 0;

          if(dflg == 0)
            for(i=0;i<nrad; i++)
              image.data[j][k][i] = ddd[i+2];
          else

            for(i=0; i<nrad; i++)
              {
                if(ddd[i+2] < 1)
                  xtmp = ddd[i+2];
                else if(ddd[i+3] < 1)
                  xtmp = 64 + ddd[i] + 2*ddd[i+1] - (3*ddd[i+2]);
                else if(ddd[i+4] < 1)
                  xtmp = 64 + ddd[i] + 2*ddd[i+1] - (3*ddd[i+3]);
                else            
                  xtmp = 64 + ddd[i] + 2*ddd[i+1] - (ddd[i+4] + 2*ddd[i+3]);
                if(xtmp < 1) xtmp = 1;
                if(xtmp > 253) xtmp = 253;
                image.data[j][k][i] = xtmp;
              }

          if(nrad < MAX_DEPTH)
            for(i=nrad; i<MAX_DEPTH; i++)
              image.data[j][k][i] = 0;

        }


      /* Adjust knum for next row */
      if(image.conn[j] == 1)
        knum = 2*knum;
      if(image.conn[j] == -1)
        knum = knum/2;
    }

  /* Zero out result array */
  for(j=0;j<(nrow+1);j++)
    for(k=0;k<ncol;k++)
      result.surface[j][k] = -1;

  edits = 0;

  printf("Finished resampling immage\n");
  printf("Exiting resamp nrow = %d\n",image.nrows);

  return true;
}
 
void vtkCellWallVisSeg::addCellEdge(double xyzv[4])
{
        // select additional edge points
        
        // conversion for legacy c code   
    int i,j,offset;
        i=(int)xyzv[0];
        j=(int)(rimage.ny-xyzv[1]-1);
        level=(int)xyzv[2];

        // from legacy c code in even_cb() ======================
        edits++;
        printf("Additional Boundary point at %d %d, level %d\n",i,j,level);
        
        flagpt(i,j,level);
        // end of legacy c code
}

void vtkCellWallVisSeg::addCellEdge3D(int x, int y)
{
        // from legacy c code in even_cb() ======================
        edits++;
        printf("Additional Boundary point %d at %d %d, elev %d\n",edits,x,y,elev);      
        flagpt6(x,y);
}

void vtkCellWallVisSeg::compute2DBoundary(int CellID)
{       
        // from legacy c code in even_cb() ======       
        // do 2d segmentation saving the result into "SURFACE result". 
        score = dynam1(); 
        
        // set the resulting boundary pixels to CellID in pixbuf
        if(score != 0) dispedge(CellID);                
        //============================================
}

void vtkCellWallVisSeg::compute3DSeg(int CellID)
{
        score = qdynam();
    dispresult(score, CellID);  
}

int vtkCellWallVisSeg::dynam1()
{
  register int i,ii,j,k,knum;
  int nrow, ncol, nrad, prad;
  int alpha[MAX_DEPTH][2][2];
  int bptr[1+MAX_COLS/2][MAX_DEPTH][2];
  int col[2];
  int score=0;
  int x;

  printf("Using .7 weighting for diaginal steps.\n");

  /* get some parameters, and initialize result structure */
  nrow = image.nrows;
  ncol = image.ncols;
  nrad = image.nrad;
  prad = image.prad;

  printf("In dynam1 nrow = %d\n",nrow);

  strcpy(result.NAME,image.NAME);
  result.nrows=image.nrows;
  result.ncols=image.ncols;
  result.nrad=image.nrad;
  for(k=0;k<nrow;k++)
    {
      result.conn[k] = image.conn[k];
      for(j=0;j<ncol;j++)
        result.surface[k][j]=-1;
    }
  result.surface[0][0] = prad;
  result.surface[nrow][0]=-1;

  /* do linear dynamic program */
  for(i=0; i<nrad; i++)
    for(j=0; j<2; j++)
      alpha[i][j][0] = 
        alpha[i][j][1] = 0;

  ii =  image.data[0][0][prad];
  if(ii == 0) 
    {
      ii = 1;
      printf("Edge point value = 0\n");
    }

  for(j=0; j<2; j++)
    {
      alpha[prad][j][0] = ii;
      col[j] = 4*j;
    }

  for(i=1;i<nrow;i++)
    {
      for(j=0;j<2;j++)
        {
          for(k=0;k<nrad;k++)
            {

              ii = image.data[i][col[j]][k];
              if(alpha[k][j][0] != 0)
                alpha[k][j][1] = alpha[k][j][0]+ii;
              bptr[i][k][j] = k;
            
              if(k != 0)
                if(alpha[k-1][j][0]!= 0)
                  {
                    x = alpha[k-1][j][0] + .7*ii;
                    if(x > alpha[k][j][1])
                      {
                        alpha[k][j][1] = x;
                        bptr[i][k][j] = k-1;
                      }
                  }

              if( k != (nrad-1))
                if(alpha[k+1][j][0] != 0)
                  {
                    x = alpha[k+1][j][0] + .7*ii;
                    if(x > alpha[k][j][1])
                      {
                        alpha[k][j][1] = x;
                        bptr[i][k][j] = k+1;
                      }
                  }
            }

          ii = 0;
          for(k=0;k<nrad;k++)
            {
              alpha[k][j][0] = alpha[k][j][1];
              alpha[k][j][1] = 0;
              if(ii < alpha[k][j][0]) ii = alpha[k][j][0] ;
            }
          if(ii == 0) 
            {
              printf("Error on first level %d hem %d\n",i,j);
              return(0);
            }
          if(image.conn[i] == 1)
            col[j] = 2*col[j];
          if(image.conn[i] == -1)
            col[j] = col[j]/2;
        }
    }

  i = nrow;

  for(k=0;k<nrad;k++)
    {
      ii = image.data[nrow][0][k];
      bptr[i][k][0] = bptr[i][k][1] = k;
      if(alpha[k][0][0] != 0)
        {
          if(alpha[k][1][0] != 0)
            {
              alpha[k][0][1] = alpha[k][0][0] + alpha[k][1][0]+ii ;
            }

          if(k != 0)
            if(alpha[k-1][1][0] != 0)
              {
                x =  alpha[k][0][0] + alpha[k-1][1][0]+.7*ii ;
                if(x > alpha[k][0][1])
                  {
                    alpha[k][0][1] = x;
                    bptr[i][k][1] = k-1;
                  }
              }
          if(k != (nrad-1))
            if(alpha[k+1][1][0] != 0)
              {
                x =  alpha[k][0][0] + alpha[k+1][1][0]+.7*ii ;
                if(x > alpha[k][0][1])
                  {
                    alpha[k][0][1] = x;
                    bptr[i][k][1] = k+1;
                  }
              }
        }

      if(k != 0)
        if(alpha[k-1][0][0] != 0)
          {
            if(alpha[k][1][0] != 0)
              {
                x =  alpha[k-1][0][0] + alpha[k][1][0]+.7*ii ;
                if(x > alpha[k][0][1])
                  {
                    alpha[k][0][1] = x;
                    bptr[i][k][0] = k-1;
                    bptr[i][k][1] = k;
                  }
              }

            if(alpha[k-1][1][0] != 0)
              {
                x =  alpha[k-1][0][0] + alpha[k-1][1][0]+.49*ii ;
                if(x > alpha[k][0][1])
                  {
                    alpha[k][0][1] = x;
                    bptr[i][k][0] = k-1;
                    bptr[i][k][1] = k-1;
                  }
              }
            if(k != (nrad-1))
              if(alpha[k+1][1][0] != 0)
                {
                  x =  alpha[k-1][0][0] + alpha[k+1][1][0]+.49*ii ;
                  if(x > alpha[k][0][1])
                    {
                      alpha[k][0][1] = x;
                      bptr[i][k][0] = k-1;
                      bptr[i][k][1] = k+1;
                    }
                }
          }

      if(k != (nrad-1))
        if(alpha[k+1][0][0] != 0)
          {
      
            if(alpha[k][1][0] != 0)
              {
                x =  alpha[k+1][0][0] + alpha[k][1][0]+.7*ii ;
                if(x > alpha[k][0][1])
                  {
                    alpha[k][0][1] = x;
                    bptr[i][k][0] = k+1;
                    bptr[i][k][1] = k;
                  }
              }

            if(k != 0)
              if(alpha[k-1][1][0] != 0)
                {
                  x =  alpha[k+1][0][0] + alpha[k-1][1][0]+.49*ii ;
                  if(x > alpha[k][0][1])
                    {
                      alpha[k][0][1] = x;
                      bptr[i][k][0] = k+1;
                      bptr[i][k][1] = k-1;
                    }
                }

            if(alpha[k+1][1][0] != 0)
              {
                x =  alpha[k+1][0][0] + alpha[k+1][1][0]+.49*ii ;
                if(x > alpha[k][0][1])
                  {
                    alpha[k][0][1] = x;
                    bptr[i][k][0] = k+1;
                    bptr[i][k][1] = k+1;
                  }
              }
          }
    }

  for(k=0;k<nrad;k++)
    alpha[k][0][0] = alpha[k][0][1];
       
  score = 0;
  knum = -1;
  for(k=0;k<nrad;k++)
    if(score < alpha[k][0][0])
      {
        score = alpha[k][0][0];
        knum = k;
      }

  if(knum == -1)
    {
      printf("ERROR: No edge found in dynam1\n");
      return(0);
    }

  k = knum;
  result.surface[nrow][0] = k;
  result.surface[0][0] = prad;

  for(j=(nrow-1);j>0;j--)
    {
      k = bptr[j+1][k][0];
      result.surface[j][0]=k;
    }

  knum = 4;
  k = result.surface[nrow][0];
  for(j=(nrow-1);j>0;j--)
    {
      k = bptr[j+1][k][1];
      result.surface[j][knum]=k;

      if(image.conn[j-1] == 1)
        knum=knum/2;
      if(image.conn[j-1] == -1)
        knum=knum*2;
    }


  printf("Leaving dynam1 nrow = %d\n",nrow);

  return ( score);
}

int vtkCellWallVisSeg::ddynam1(int i, int knum)
{
  register int j,k;
  int ihm,kl,kll,ku,ks;
  int nrad, tscore;
  int alpha [9][MAX_COLS];
  int bptr [9][MAX_COLS];
  int a1,a3;
  int b1, b2, b3, b4, m;
  int c1, c2, kk;

  /* get some parameters */
  nrad = image.nrad;

  if((image.conn[i-1] !=-1) || (image.conn[i] != 0))
    {
      printf("ERROR:  Called ddynam1 with incorrect conditions %d %d\n",
             image.conn[i-1], image.conn[i]);
      return 0;
    }

  for(ihm=0; ihm<2; ihm++)
    {
      kl = (knum/2)*ihm;
      kll = kl*2;
      ku = (knum/2)*(1-ihm);

      tscore = 0;

      for(k=0;k<9;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;
              
      c1 = result.surface[i][kl];
      c2 = result.surface[i+1][kl];
      kk = 3*(c1 - result.surface[i-1][kll]+1) + (c2-c1+1);

      alpha[kk][0] = image2.data[i][kl][c1] + image2.data[i+1][kl][c2];
              
      for(j=1;j<(knum/2);j++)
        for(k=0; k<9; k++)
          if(alpha[k][j-1] != 0)
            {
              b1 = k/3;
              b1 = result.surface[i-1][(kl+j-1)*2] + b1 -1;
              b2 = k%3;
              b2 = b1+b2-1;
              if((b1>-1) && (b1<nrad) && (b2>-1) && (b2<nrad))
                for(m=0;m<9;m++)
                  {
                    b3=m/3;
                    b3=result.surface[i-1][(kl+j)*2] + b3-1;
                    b4 = m%3;
                    b4 = b3+b4-1;
                    if((b3>-1) && (b3<nrad) && 
                       (b4>-1) && (b4<nrad) &&
                       (abs(b3-b1)<2) && (abs(b4-b2)<2)
                       && (abs(b3-result.surface[i-1][(kl+j)*2+1])<2))
                      {
                        a3 = alpha[k][j-1] +
                          image2.data[i][kl+j][b3] +
                          image2.data[i+1][kl+j][b4];
                            
                        if(alpha[m][j] < a3)
                          {
                            alpha[m][j] = a3;
                            bptr[m][j] = k;
                          }
                      }
                  }
            }
                  
      for(m=0;m<9;m++)
        {
          b3=m/3;
          ks = kl + (knum/2) -1;
          b3=result.surface[i-1][ks*2] + b3-1;
          b4 = m%3;
          b4 = b3+b4-1;
          if((b3>-1) && (b3<nrad) &&
             (b4>-1) && (b4<nrad) &&
             (abs(b3-result.surface[i][ku])<2) &&
             (abs(b4-result.surface[i+1][ku])<2) &&
             (alpha[m][knum/2-1] > tscore))
            {
              tscore = alpha[m][knum/2-1];
              a1 = m;
              result.surface[i][ks] = b3;
              result.surface[i+1][ks] = b4;
              for(j=(knum/2-1);j>0;j--)
                {
                  a1 = bptr[a1][j];
                  b3=a1/3;
                  b3=result.surface[i-1][(kl+j-1)*2] + b3-1;
                  b4 = a1%3;
                  b4 = b3+b4-1;
                  result.surface[i][kl+j-1] = b3;
                  result.surface[i+1][kl+j-1] = b4;
                }
            }
        }
 
      if(tscore == 0)
        {
          printf("ERROR: Can not compute a post reduction layer. i=%d on hemisphere %d \n",i,ihm);
          return(0);
        }
  }

  return(tscore);
}

void vtkCellWallVisSeg::dispedge(int id)
{
  int i,j;
  int o1x, o1y, o2x, o2y;

  int nrow,knum,nrad, ii,jj;
  double dm, dis;
  double basevec[3], baservec[3], basecvec[3], uvec[3];
  double pi=3.14159269;
  double x, minrat, maxrat;
  int cent[3], bound[3];
  double y,yy;
  int offset;

  nrow = image.nrows;
  nrad = image.nrad;
  minrat = image.minrat;
  maxrat = image.maxrat;

  printf("In dispedge. nrow = %d\n",nrow);

  cent[0] = rimage.cent[0];
  cent[1] = rimage.cent[1];
  cent[2] = rimage.cent[2];
  bound[0] = rimage.bound[0];
  bound[1] = rimage.bound[1];
  bound[2] = rimage.bound[2];

  dis = (bound[0]-cent[0])*(bound[0]-cent[0]) + 
    (bound[1]-cent[1])*(bound[1]-cent[1]);
  dis = sqrt(dis);
  
  dm = maxrat - minrat;
  dm = dm/nrad;
  
  for(i=0;i<3;i++)
    basevec[i] = (bound[i] - cent[i])/dis;

  offset = levelsize*level + (rimage.nx)*3*bound[1] + bound[0]*3;
  pixbuf[offset] = id;
  pixbuf[offset+1] = pixbuf[offset+2] = 0;

  o1x = o2x = bound[0];
  o1y = o2y = bound[1];

  knum = 4;
  for(j=1;j<nrow;j++)
    {

      /* generate base unit vector for this row */
      x = j*pi/nrow;

      baservec[0] = basevec[0]*cos(x);
      baservec[1] = basevec[1]*cos(x);
      basecvec[0] = -basevec[1]*sin(x);
      basecvec[1] =  basevec[0]*sin(x);
      uvec[0] = baservec[0] + basecvec[0];
      uvec[1] = baservec[1] + basecvec[1];
      x = dis*(minrat+dm*result.surface[j][0]);
      y = uvec[0]*x + cent[0];
      yy = uvec[1]*x + cent[1];
      ii = y+.5;
      jj = yy +.5;
      offset = levelsize*level + (rimage.nx)*3*jj + ii*3; 
      pixbuf[offset] = id;
      pixbuf[offset+1] = pixbuf[offset+2] = 0;


      if((abs(ii - o1x) > 1) || (abs(jj - o1y) > 1))
        {
          o1x = (o1x + ii)/2;
          o1y = (o1y + jj)/2;
          offset = levelsize*level + (rimage.nx)*3*o1y + o1x*3; 
          pixbuf[offset] = id;
          pixbuf[offset+1] = pixbuf[offset+2] = 0;
        }
      o1x = ii;
      o1y = jj;

      uvec[0] = baservec[0] - basecvec[0];
      uvec[1] = baservec[1] - basecvec[1];
          
      x = dis*(minrat+dm*result.surface[j][knum]);
      y = uvec[0]*x + cent[0];
      yy = uvec[1]*x + cent[1];
      ii = y+.5;
      jj = yy +.5;

      offset = levelsize*level + (rimage.nx)*3*jj + ii*3; 
      pixbuf[offset] = id;
      pixbuf[offset+1] = pixbuf[offset+2] = 0;

      if((abs(ii - o2x) > 1) || (abs(jj - o2y) > 1))
        {
          o2x = (o2x + ii)/2;
          o2y = (o2y + jj)/2;
          offset = levelsize*level + (rimage.nx)*3*o2y + o2x*3; 
          pixbuf[offset] = id;
          pixbuf[offset+1] = pixbuf[offset+2] = 0;
        }

      o2x = ii;
      o2y = jj;

      /* Adjust knum for next row */
      if(image.conn[j] == 1)
        knum = 2*knum;
      if(image.conn[j] == -1)
        knum = knum/2;
    }

  uvec[0] = -basevec[0];
  uvec[1] = -basevec[1];
  x = dis*(minrat+dm*result.surface[nrow][0]);
  y = uvec[0]*x + cent[0];
  yy = uvec[1]*x + cent[1];
  ii = y+.5;
  jj = yy +.5;
  offset = levelsize*level + (rimage.nx)*3*jj + ii*3; 
  pixbuf[offset] = id;
  pixbuf[offset+1] = pixbuf[offset+2] = 0;

  if((abs(ii - o1x) > 1) || (abs(jj - o1y) > 1))
    {
      o1x = (o1x + ii)/2;
      o1y = (o1y + jj)/2;
      offset = levelsize*level + (rimage.nx)*3*o1y + o1x*3; 
      pixbuf[offset] = id;
      pixbuf[offset+1] = pixbuf[offset+2] = 0;
    }
  if((abs(ii - o2x) > 1) || (abs(jj - o2y) > 1))
    {
      o2x = (o2x + ii)/2;
      o2y = (o2y + jj)/2;
      offset = levelsize*level + (rimage.nx)*3*o2y + o2x*3; 
      pixbuf[offset] = id;
      pixbuf[offset+1] = pixbuf[offset+2] = 0;
    }

}

int vtkCellWallVisSeg::qdynam()
{
  register int i,j,ihm,k,knum,ii,jj;
  int nrow, ncol, nrad, prad, srad;
  int alpha [MAX_DEPTH][MAX_COLS];
  int bptr [MAX_DEPTH][MAX_COLS];
  int kl, ku, kk, a1, a2, a3, tscore, score;
  int b1, b2, b3, m;
  int c1, c2, c3, c4;
  int d1, d2,d3;
  int jjmin, jjmax;

  /* get some parameters */
  nrow = image.nrows;
  ncol = image.ncols;
  nrad = image.nrad;
  prad = image.prad;
  srad = result.surface[nrow][0];

  /* Copy image to image2 */
  for(i=0; i<= nrow; i++)
    for(j=0;j<ncol;j++)
      for(k=0;k<nrad;k++)
        image2.data[i][j][k] = image.data[i][j][k];

  /* Do the south pole scores first */
 
  score = image2.data[nrow][0][srad];

  /* Do the first row as special code */
  for(ihm=0;ihm<2;ihm++)
    {
      for(k=0;k<nrad;k++)
        for(i=0;i<4;i++)
          alpha[k][i]=0;

      kl = srad-1;
      if(kl < 0) kl = 0;
      ku = srad+2;
      if(ku > nrad) ku = nrad;

      tscore = 0;
      kk = result.surface[nrow-1][4*ihm];
      alpha[kk][0] = image2.data[nrow-1][4*ihm][kk];

      for(j=1;j<4;j++)
        for(k=kl; k<ku; k++)
          for(a1=-1;a1<2;a1++)
            {
              a2 = k+a1;
              if((a2>-1) && (a2<nrad))
                if(alpha[a2][j-1] != 0)
                  {
                    a3 = alpha[a2][j-1] + image2.data[nrow-1][j][k];
                    if(alpha[k][j] < a3)
                      {
                        alpha[k][j] = a3;
                        bptr[k][j] = a2;
                      }
                  }
            }
      kk = result.surface[nrow-1][4-4*ihm];
      for(k=kl; k<ku; k++)
        if(abs(kk-k) < 2)
          if(alpha[k][3] > tscore)
            {
              tscore = alpha[k][3];
              a1 = k;
              result.surface[nrow-1][3+4*ihm]=k;
              for(j=3; j>0; j--)
                {
                  a1=bptr[a1][j];
                  result.surface[nrow-1][j+4*ihm-1] = a1;
                }
            }
      score = score + tscore;

    }

  /* now for the other rows */
  knum = 16;

  /* for each row */
  for(i=nrow-2;i >= image.reduce;i--)
    {

      if((i%10) == 0) printf("Row %d of %d\n",i,nrow);
      for(k=0;k<nrad;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;


      /* Post expanding row. Double step */
      if(image.conn[i] == -1)
        {
          tscore = ddynams4(i,knum);
          if(tscore ==0) return(-i);

        }

      /* Expanding row. Double step */
      else if(image.conn[i-1] == -1)
        {
          tscore = ddynams3(i,knum);
          if(tscore ==0) return(-i);

        }

      /* Constant row */
      if((image.conn[i] ==0) && (image.conn[i-1] == 0))
        {
          if((image.conn[i-2] ==0) && (image.conn[i-3] == 0))
            tscore = qdynams2(i,knum);
          else if((image.conn[i+1] ==0))
            {
              tscore = 1;
              i--;
            }
          else
            tscore = ddynams2(i,knum);

          if(tscore ==0) return(-i);
        }

      score = score + tscore;

      /* Adjust knum for next row */
      if(image.conn[i-1] == -1)
        knum = 2*knum;

    }


  /*  Put in attraction cones */

  for(ihm=0;ihm<2;ihm++)
    {
      kl=1+knum*ihm/2;
      ku = (knum/2) + knum*ihm/2; 
      for(i=kl;i<ku;i++)
        {
          jjmin =  result.surface[image.reduce][i];
          jjmax =  result.surface[image.reduce][i] +1;
          for(jj=image.reduce-1; jj > (nrow-image.reduce);jj--)
            {
              jjmin = (jjmin>0) ? jjmin-1 : 0;
              jjmax = (jjmax<nrad) ? jjmax+1 : nrad;

              for(ii=jjmin; ii<jjmax; ii++)
                if(image2.data[jj][i][ii]> 0)
                  image2.data[jj][i][ii] += 6400;

            }
        }
    }



  /* Now start with the north pole */  
  score += image2.data[0][0][prad];

  /* Do the first row as special code */
  for(ihm=0;ihm<2;ihm++)
    {
      for(k=0;k<nrad;k++)
        for(i=0;i<4;i++)
          alpha[k][i]=0;

      kl = prad-1;
      if(kl < 0) kl = 0;
      ku = prad+2;
      if(ku > nrad) ku = nrad;

      tscore = 0;
      kk = result.surface[1][4*ihm];
      alpha[kk][0] = image2.data[1][4*ihm][kk];

      for(j=1;j<4;j++)
        for(k=kl; k<ku; k++)
          for(a1=-1;a1<2;a1++)
            {
              a2 = k+a1;
              if((a2>-1) && (a2<nrad))
                if(alpha[a2][j-1] != 0)
                  {
                    a3 = alpha[a2][j-1] + image2.data[1][j][k];
                    if(alpha[k][j] < a3)
                      {
                        alpha[k][j] = a3;
                        bptr[k][j] = a2;
                      }
                  }
            }
      kk = result.surface[1][4-4*ihm];
      for(k=kl; k<ku; k++)
        if(abs(kk-k) < 2)
          if(alpha[k][3] > tscore)
            {
              tscore = alpha[k][3];
              a1 = k;
              result.surface[1][3+4*ihm]=k;
              for(j=3; j>0; j--)
                {
                  a1=bptr[a1][j];
                  result.surface[1][j+4*ihm-1] = a1;
                }
            }
      score = score + tscore;
    }


  /* now for the other rows */
  knum = 16;

  /* for each row */
  for(i=2;i<(image.reduce-3);i++)
    {

      if((i%10) == 0) printf("Row %d of %d\n",i,nrow);
      for(k=0;k<nrad;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;

      /* Post expanding row. Double step */
      if(image.conn[i-1] ==1)
        {
          tscore = ddynam4(i,knum);
          if(tscore ==0) return(-i);

        }

      /* Expanding row. Double step */
      if(image.conn[i] ==1)
        {
          tscore = ddynam3(i,knum);
          if(tscore ==0) return(-i);

        }


     /* Reduce row. Double step */
      if(image.conn[i] == -1)
        {
          tscore = ddynam0(i,knum);
          if(tscore ==0) return(-i);

        }

      /* Post Reduce row. Double step */
      if(image.conn[i-1] == -1)
        {
          tscore = ddynam1(i,knum);
          if(tscore ==0) return(-i);

        }

      /* Constant row */
      if((image.conn[i-1] ==0) && (image.conn[i] == 0))
        {
          if((image.conn[i+1] ==0) && (image.conn[i+2] == 0))
            tscore = qdynam2(i,knum);
          else if((image.conn[i+1] ==0) && (image.conn[i+2] == -1))
            {
              tscore = qdynam0(i,knum);
              i += 2;
            }
          else if((image.conn[i-2] ==0))
            {
              tscore = 1;
              i++;
            }
          else
              tscore = ddynam2(i,knum);

          if(tscore ==0) return(-i);
        }

      score = score + tscore;

      /* Adjust knum for next row */
      if(image.conn[i] == 1)
        knum = 2*knum;

      if(image.conn[i] == -1)
        knum = knum/2;
    }


  /*  Test for merge of north/south. */
  jj = 0;

  for(j=0;j<ncol;j=j+16)
    {
      for(i=0;i<16;i++)
        {
          if(abs(result.surface[image.reduce-1][i+j]
                 - result.surface[image.reduce][i+j]) > 1) 
            {
              jj++;
              printf("%d %d %d %d %d %d\n",jj,i+j,
                     result.surface[image.reduce-1][i+j],
                     result.surface[image.reduce][i+j],
                     image2.data[image.reduce-1][i+j][result.surface[image.reduce-1][i+j]],
                     image2.data[image.reduce-1][i+j][result.surface[image.reduce][i+j]]);
            }
        }
    }

  if(jj > 0)
    printf("ERROR in merge at %d points\n",jj);

  printf("Exit qdynam, score = %d\n",score);
  return ( score);
}

void vtkCellWallVisSeg::dispresult(int numb, int CellID) //CellID added by jusub
{

  int i,j,k;
  int zz;

  int nrow,knum,nrad, ii,jj,kk, nrowp;
  double dm, dis, dis2;
  double basevec[3], baservec[3], basecvec[3], uvec[3];
  double pi=3.14159269;
  double x, minrat, maxrat;
  int cent[3], bound[3];
  double y,yy;
  int offset;
  int nx,ny,nz;

  nrow = image.nrows;
  nrad = image.nrad;
  minrat = image.minrat;
  maxrat = image.maxrat;

  nx = rimage.nx;
  ny = rimage.ny;
  nz = rimage.nz;


  cent[0] = rimage.cent[0];
  cent[1] = rimage.cent[1];
  cent[2] = rimage.cent[2];
  bound[0] = rimage.bound[0];
  bound[1] = rimage.bound[1];
  bound[2] = rimage.bound[2];

  dis = (bound[0]-cent[0])*(bound[0]-cent[0]) + 
    (bound[1]-cent[1])*(bound[1]-cent[1]);
  dis = sqrt(dis);

  dm = maxrat - minrat;
  dm = dm/nrad;

  
  for(i=0;i<3;i++)
    basevec[i] = (bound[i] - cent[i])/dis;

  knum = 8;

  if(numb > 0)
    nrowp = nrow;
  else
    nrowp = -numb;

  for(j=1;j<nrowp;j++)
    {

      /* generate base unit vector for this row */
      x = j*pi/nrow;

      baservec[0] = basevec[0]*cos(x);
      baservec[1] = basevec[1]*cos(x);
      baservec[2] = 0;
      basecvec[0] = -basevec[1]*sin(x);
      basecvec[1] =  basevec[0]*sin(x);
      basecvec[2] = 0;

      dis2 = basecvec[0]*basecvec[0] + basecvec[1]*basecvec[1];
      dis2 = sqrt(dis2);

      /* for each col in this row */
      for(k=0;k<knum;k++)
        {
          
          /* generate unit vector for this scan line */
          y = (k*pi*2)/knum;
          uvec[0] = baservec[0] + basecvec[0] * cos(y);
          uvec[1] = baservec[1] + basecvec[1] * cos(y);
          uvec[2] = dis2*sin(y)/rimage.aspratio;
          
          x = dis*(minrat+dm*result.surface[j][k]);
          y = uvec[0]*x + cent[0];
          yy = uvec[1]*x + cent[1];
          zz = uvec[2]*x + cent[2];

          ii = y+.5;
          jj = yy +.5;
          kk = zz + .5;

          if((ii<nx) && (jj < ny) && (kk < nz)) 
            {
              offset = levelsize*kk + (rimage.nx)*3*jj + 3*ii;

              if(numb > 0)
                {
                  pixbuf[offset] = CellID;
                  pixbuf[offset+1] = pixbuf[offset+2] = 0;
                }
              else
                {
                  pixbuf[offset+1] = CellID;
                  pixbuf[offset] = pixbuf[offset+2] = 0;
                }
            }
        }


      /* Adjust knum for next row */
      if(image.conn[j] == 1)
        knum = 2*knum;
      if(image.conn[j] == -1)
        knum = knum/2;
    }

  uvec[0] = -basevec[0];
  uvec[1] = -basevec[1];
  x = dis*(minrat+dm*result.surface[nrow][0]);
  y = uvec[0]*x + cent[0];
  yy = uvec[1]*x + cent[1];
  ii = y+.5;
  jj = yy +.5;
  kk = cent[2];

  offset = levelsize*kk + (rimage.nx)*3*jj + 3*ii;
  pixbuf[offset] = CellID;
  pixbuf[offset+1] = pixbuf[offset+2] = 0;

  return;
}

void vtkCellWallVisSeg::flagpt(int i, int j, int k)
{
  int nrow,knum,nrad,jj, kk;
  double dm, dis, dis2;
  double pi=M_PI;
  double x, y, minrat, maxrat;
  int cent[3], bound[3];
  int row;

  nrow = image.nrows;
  nrad = image.nrad;
  minrat = image.minrat;
  maxrat = image.maxrat;

  cent[0] = rimage.cent[0];
  cent[1] = rimage.cent[1];
  cent[2] = rimage.cent[2];
  bound[0] = rimage.bound[0];
  bound[1] = rimage.bound[1];
  bound[2] = rimage.bound[2];

  if(k != cent[2])
    {
      printf("Added point not on same level as center.  Thus will have\n");
      printf("no effect on first level boundary.  Center is on level %d\n",
             cent[2]);
      return;
    }

  dis = (bound[0]-cent[0])*(bound[0]-cent[0]) + 
    (bound[1]-cent[1])*(bound[1]-cent[1]);
  dis = sqrt(dis);

  dis2 = (i-cent[0])*(i-cent[0]) +
    (j-cent[1])*(j-cent[1]);
  dis2 = sqrt(dis2);

  x = (i-cent[0])*(bound[0]-cent[0]) + (j-cent[1])*(bound[1]-cent[1]);
  x = x/(dis*dis2);
  row = (acos(x)*nrow/pi) + .5;

  dm = maxrat - minrat;
  dm = dm/nrad;

  kk = ((dis2/dis) - minrat)/dm + .5;

  y = (i-cent[0])*(bound[1]-cent[1]) - (j-cent[1])*(bound[0]-cent[0]);
  if(y <= 0.0)
      knum = 0;
  else
    {
      knum = 4;
      for(jj=1;jj<row;jj++)
        {
          if(image.conn[jj] == 1)
            knum = 2*knum;
          if(image.conn[jj] == -1)
            knum = knum/2;
        }
    }

  if((kk > nrad) || (kk<0))
    printf("Point not in range of immage. %d %d\n",kk,nrad);
  else
    if(image.data[row][knum][kk]> 0)
      image.data[row][knum][kk] = 6400;

}

int vtkCellWallVisSeg::interp( double pt[3])
{

  /* Will take a double pt in 3-space and interprilate between the
     int points for the value for this point.
  */

  double frac[3], val, mfrac[3];
  int i,j, ipt[3];

  /*  for(j=0;j<3;j++)
    printf("%d %f\n",j,pt[j]);
  */

  /* Is the point inside the space? */
  if((pt[0] > (rimage.nx-1)) || (pt[1] > (rimage.ny-1)) || 
     (pt[2] > (rimage.nz-1)))
    return(1);
  for(i=0; i<3; i++)
    if(pt[i] < 0)
      return(1);

  for(i=0; i<3; i++)
    {
      j = pt[i];
      frac[i] = pt[i] - j;
      mfrac[i] = 1.0 - frac[i];
      ipt[i] = j;
    }

  val = 0.5;
  val += mfrac[0]*mfrac[1]*mfrac[2]*rimage.data[ipt[0]][ipt[1]][ipt[2]];
  val += frac[0]*mfrac[1]*mfrac[2]*rimage.data[ipt[0]+1][ipt[1]][ipt[2]];
  val += mfrac[0]*frac[1]*mfrac[2]*rimage.data[ipt[0]][ipt[1]+1][ipt[2]];
  val += frac[0]*frac[1]*mfrac[2]*rimage.data[ipt[0]+1][ipt[1]+1][ipt[2]];
  val += mfrac[0]*mfrac[1]*frac[2]*rimage.data[ipt[0]][ipt[1]][ipt[2]+1];
  val += frac[0]*mfrac[1]*frac[2]*rimage.data[ipt[0]+1][ipt[1]][ipt[2]+1];
  val += mfrac[0]*frac[1]*frac[2]*rimage.data[ipt[0]][ipt[1]+1][ipt[2]+1];
  val += frac[0]*frac[1]*frac[2]*rimage.data[ipt[0]+1][ipt[1]+1][ipt[2]+1];

  j = val;

  /* If closest point is 0, set this spherical point to 0 */
  for(i=0; i<3; i++)
    ipt[i] = pt[i]+.5;
  if(rimage.data[ipt[0]][ipt[1]][ipt[2]] <1) j=0;


  if(j<1)
    {
      if((zflg == 1) && (dflg != 1))
        j=-6400;
      else j = 1;
    } 
  return(j);
}

void vtkCellWallVisSeg::flagpt6(int iii, int jjj)
{

  int i,j,ii,jj;
  int icol[MAX_COLS];
  int ncol, nrow, knum, nrad;
  int rad, ang;
  double maxrat, minrat;
  double dx, dm, dxx, dmm, dd, di, dj, rang;
  int k;
  int col2, jjmin, jjmax, rr, rrr;
  /* the icol table contains the col position, given the current elevation */

  nrow = image.nrows;
  ncol = image.ncols;
  nrad = image.nrad;
  minrat = image.minrat;
  maxrat = image.maxrat;

  icol[0] = 0;
  icol[nrow] = 0;
  knum = 8;
  for(i=1; i<nrow; i++)
    {
      ii = (elev * knum + .5)/ncol;
      icol[i] = ii;
      ii += knum/2;
      if(ii >= knum) ii -= knum;
      icol[i+nrow] = ii;
    
      if(image.conn[i] == 1)
        knum = 2*knum;

      if(image.conn[i] == -1)
        knum = knum/2;
    }

  /* Compute the distance to the center */
  dx = maxrat* MAX_DEPTH/(maxrat - minrat);
  dm = dx * minrat  / maxrat;

  dd = dx;

  dxx = dx*dx;
  dmm = dm * dm;

  j = jjj;

  dj = j-dd;
  dj = dj*dj;

  i = iii;

  di = i-dd;
  di = di*di + dj;
  if(di > dxx)
    printf("Point outside image area.\n");

  else if(di < dmm)
    printf("Point too close to image center.\n");

  else
    {
      rad = .5 + sqrt(di) - dm;
      rang = atan((i-dd)/(pwidth - (j+dd))) * nrow / M_PI ;
      if((dd+j)>pwidth) rang = rang + nrow;
      if(rang < 0.0) rang = rang + ncol;
      ang = .5 + rang;
      if(ang >= ncol) ang = ang - ncol;
      k = ang;
      if(k > nrow) k = ncol - k;
      if(image.data[k][icol[ang]][rad] > 0)
         image.data[k][icol[ang]][rad] += 6400;

      printf("K %d image.reduce %d\n",k,image.reduce);

      /* funnel for NtoS direction */
      if(k < image.reduce)
        {

          col2 = icol[ang];
          jjmin = rad;
          jjmax = rad;
          rr = k;
          rrr = (k>42)? 40:(k-2);
          for(jj=1;jj<rrr;jj++)
            {
              rr = rr-1;
              if(image.conn[rr] == 1)
                col2 = col2/2;
              if(image.conn[rr] == -1)
                col2 = 2*col2;
              jjmin = (jjmin>0) ? jjmin-1 : 0;
              jjmax = (jjmax<nrad) ? jjmax+1 : nrad;
              
              for(ii=jjmin; ii<jjmax; ii++)
                if( image.data[rr][col2][ii] > 0)
                  image.data[rr][col2][ii] += 6400;
            }
        }

      /* funnel for StoN direction */
      if(k > (image.reduce-40))
        {

          col2 = icol[ang];
          jjmin = rad;
          jjmax = rad;
          rr = k;
          rrr = (k>(nrow-42)) ? 40:(nrow-(k-2));
          for(jj=1;jj<rrr;jj++)
            {
              rr = rr+1;
              if(image.conn[rr-1] == -1)
                col2 = col2/2;
              if(image.conn[rr-1] == 1)
                col2 = 2*col2;
              jjmin = (jjmin>0) ? jjmin-1 : 0;
              jjmax = (jjmax<nrad) ? jjmax+1 : nrad;

              for(ii=jjmin; ii<jjmax; ii++)
                if( image.data[rr][col2][ii] > 0)
                  image.data[rr][col2][ii] += 6400;
            }
        }
    }
  return;
}

int vtkCellWallVisSeg::qdynam0(int i, int knum)
{
  register int j,k;
  int ihm,kk,kl,ku,jj;
  int nrad, tscore;
  int alpha [81][MAX_COLS];
  int bptr [81][MAX_COLS];
  int a1,a3;
  int b1, b2, b3, b4, m;
  int c1, c2, c3, c4;
  int d1, d2, d3, d4;

  /* get some parameters */
  nrad = image.nrad;
  if((image.conn[i-1] !=0) || (image.conn[i] != 0) ||
     (image.conn[i+1] !=0) || (image.conn[i+2] != -1) )
    {
      printf("ERROR:  Called qdynam0 with incorrect conditions %d %d %d %d\n",
             image.conn[i-1], image.conn[i], image.conn[i+1], image.conn[i+2]);
      exit(9);
    }

  for(ihm=0;ihm<2;ihm++)
    {
      kl = (knum/2)*ihm;
      ku = (1-ihm)*(knum/2);

      tscore = 0;

      for(k=0;k<81;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;
              
      c1 = result.surface[i][kl];
      c2 =  result.surface[i+1][kl];
      c3 =  result.surface[i+2][kl];
      c4 =  result.surface[i+3][kl/2];
      kk = 27*(c1-result.surface[i-1][kl]+1) + 9*(c2-c1+1) + 3*(c3-c2+1) +
        (c4-c3+1);

      alpha[kk][0] = image2.data[i][kl][c1] + image2.data[i+1][kl][c2] +
            image2.data[i+2][kl][c3] + image2.data[i+3][kl/2][c4];
              
      for(j=1;j<(knum/2);j++)
        for(k=0; k<81; k++)
          if(alpha[k][j-1] != 0)
            {
              b1 = k/27;
              b1 = result.surface[i-1][j+kl-1] + b1 -1;
              b2 = (k%27)/9;
              b2 = b1+b2-1;
              b3 = (k%9)/3;
              b3 = b2 + b3 -1;
              b4 = k%3;
              b4 = b3 + b4 -1;
              if((b1>-1) && (b1<nrad) && (b2>-1) && (b2<nrad) &&
                 (b3>-1) && (b3<nrad) && (b4>-1) && (b4<nrad))
                for(m=0;m<81;m++)
                  {
                    d1 = m/27;
                    d1=result.surface[i-1][j+kl] + d1-1;
                    d2 = (m%27)/9;
                    d2 = d1 + d2 -1;
                    d3 = (m%9)/3;
                    d3 = d2 + d3 -1;
                    d4 = m%3;
                    d4 = d3 + d4 -1;
                    
                    if((d1>-1) && (d1<nrad) &&
                       (d2>-1) && (d2<nrad) &&
                       (d3>-1) && (d3<nrad) && 
                       (d4>-1) && (d4<nrad) &&
                       (abs(b1-d1)<2) && (abs(b2-d2)<2) &&
                       (abs(b3-d3)<2))
                      {
                        a3 =0;
                        if(((j%2) == 1) && (d4 == b4))
                          a3 = alpha[k][j-1] +
                            image2.data[i][j+kl][d1] +
                            image2.data[i+1][j+kl][d2] +
                            image2.data[i+2][j+kl][d3];
                        if(((j%2) == 0) && (abs(d4-b4) < 2))
                          a3 = alpha[k][j-1] +
                            image2.data[i][j+kl][d1] +
                            image2.data[i+1][j+kl][d2] +
                            image2.data[i+2][j+kl][d3] +
                            image2.data[i+3][(j+kl)/2][d4];
                            
                        if(alpha[m][j] < a3)
                          {
                            alpha[m][j] = a3;
                            bptr[m][j] = k;
                          }
                      }
                  }
            }

      jj=(knum/2)-1;

      for(m=0;m<81;m++)
        {
          d1 = m/27;
          d1=result.surface[i-1][kl + jj] + d1-1;
          d2 = (m%27)/9;
          d2 = d1 + d2 -1;
          d3 = (m%9)/3;
          d3 = d2 + d3 -1;
          d4 = m%3;
          d4 = d3 + d4 -1;

          if((d1>-1) && (d1<nrad) &&
             (d2>-1) && (d2<nrad) &&
             (d3>-1) && (d3<nrad) &&
             (d4>-1) && (d4<nrad) &&
             (abs(result.surface[i][ku]-d1)<2) && 
             (abs(result.surface[i+1][ku]-d2)<2) &&
             (abs(result.surface[i+2][ku]-d3)<2) && 
             (abs(result.surface[i+3][ku/2]-d4)<2) &&
             (alpha[m][jj] > tscore))
            {
              tscore = alpha[m][jj];
              a1 = m;

              result.surface[i][jj+kl] = d1;
              result.surface[i+1][jj+kl] = d2;
              result.surface[i+2][jj+kl] = d3;
              result.surface[i+3][(jj+kl)/2] = d4;
              for(j=jj;j>0;j--)
                {
                  a1 = bptr[a1][j];
                  d1 = a1/27;
                  d1=result.surface[i-1][kl+j-1] + d1-1;
                  d2 = (a1%27)/9;
                  d2 = d1 + d2 -1;
                  d3 = (a1%9)/3;
                  d3 = d2 + d3 -1;
                  d4 = a1%3;
                  d4 = d3 + d4 -1;

                  result.surface[i][kl+j-1] = d1;
                  result.surface[i+1][kl+j-1] = d2;
                  result.surface[i+2][kl+j-1] = d3;
                  result.surface[i+3][(kl+j-1)/2] = d4;
                }
            }
        }
    
      if(tscore == 0)
        {
          printf("ERROR: Can not compute a quad reduction layer i=%d Hem = %d\n"
                 ,i,ihm);

          for(j=0;j<knum;j++)
            {
              printf("%d ",result.surface[i-1][j]);
              if(j%16 == 15) printf("\n");
            }
          k = knum/2;
          for(j=0;j<3;j++)
            printf("%d %d %d\n",j,result.surface[i+j][0],
                   result.surface[i+j][k]);
          printf("%d %d %d\n",3,result.surface[i+3][0],
                 result.surface[i+3][k/2]);

          return(0);
        }
    }
  return(tscore);
}

int vtkCellWallVisSeg::qdynam2(int i, int knum)
{
  register int j,k;
  int ihm,kk,kl,ku,jj;
  int nrad, tscore;
  int alpha [81][MAX_COLS];
  int bptr [81][MAX_COLS];
  int a1,a3;
  int b1, b2, b3, b4, m, m1, m2, m3, m4;
  int c1, c2, c3, c4;
  int d1, d2, d3, d4;

  /* get some parameters */
  nrad = image.nrad;
  if((image.conn[i-1] !=0) || (image.conn[i] != 0) ||
     (image.conn[i+1] !=0) || (image.conn[i+2] != 0) )
    {
      printf("ERROR:  Called qdynam2 with incorrect conditions %d %d %d %d\n",
             image.conn[i-1], image.conn[i], image.conn[i+1], image.conn[i+2]);
      exit(9);
    }

  for(ihm=0;ihm<2;ihm++)
    {
      kl = (knum/2)*ihm;
      ku = (1-ihm)*(knum/2);

      tscore = 0;

      for(k=0;k<81;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;
              
      c1 = result.surface[i][kl];
      c2 =  result.surface[i+1][kl];
      c3 =  result.surface[i+2][kl];
      c4 =  result.surface[i+3][kl];
      kk = 27*(c1-result.surface[i-1][kl]+1) + 9*(c2-c1+1) + 3*(c3-c2+1) +
        (c4-c3+1);

      alpha[kk][0] = image2.data[i][kl][c1] + image2.data[i+1][kl][c2] +
            image2.data[i+2][kl][c3] + image2.data[i+3][kl][c4];
              
      for(j=1;j<(knum/2);j++)
        for(k=0; k<81; k++)
          if(alpha[k][j-1] != 0)
            {
              b1 = k/27;
              b1 = result.surface[i-1][j+kl-1] + b1 -1;
              b2 = (k%27)/9;
              b2 = b1+b2-1;
              b3 = (k%9)/3;
              b3 = b2 + b3 -1;
              b4 = k%3;
              b4 = b3 + b4 -1;
              if((b1>-1) && (b1<nrad) && (b2>-1) && (b2<nrad) &&
                 (b3>-1) && (b3<nrad) && (b4>-1) && (b4<nrad))
                for(m1=-1;m1<2;m1++)
                  {
                    d1=result.surface[i-1][j+kl] + m1;
                    if((d1>-1) && (d1<nrad) && (abs(b1-d1)<2))
                      {
                        for(m2 =-1;m2<2;m2++)
                          {
                            d2 = d1+m2;
                            if((d2>-1) && (d2<nrad) && (abs(b2-d2)<2))
                              {
                                for(m3 =-1;m3<2;m3++)
                                  {
                                    d3 = d2 + m3;
                                    if((d3>-1) && (d3<nrad) && (abs(b3-d3)<2))
                                      {
                                        for(m4 =-1;m4<2;m4++)
                                          {
                                            d4 = d3 + m4;
                                            if((d4>-1) && (d4<nrad) && 
                                               (abs(b4-d4)<2))
                                              {
                                                m = 27*(m1+1) + 9*(m2+1) 
                                                  + 3*(m3+1) +m4+1;
                        a3 = alpha[k][j-1] +
                          image2.data[i][j+kl][d1] +
                          image2.data[i+1][j+kl][d2] +
                          image2.data[i+2][j+kl][d3] +
                          image2.data[i+3][j+kl][d4];
                            
                        if(alpha[m][j] < a3)
                          {
                            alpha[m][j] = a3;
                            bptr[m][j] = k;
                          }
                      }
                  }
            }
                                  }}}}}}

      jj=(knum/2)-1;

      for(m=0;m<81;m++)
        {
          d1 = m/27;
          d1=result.surface[i-1][kl + jj] + d1-1;
          d2 = (m%27)/9;
          d2 = d1 + d2 -1;
          d3 = (m%9)/3;
          d3 = d2 + d3 -1;
          d4 = m%3;
          d4 = d3 + d4 -1;

          if((d1>-1) && (d1<nrad) &&
             (d2>-1) && (d2<nrad) &&
             (d3>-1) && (d3<nrad) &&
             (d4>-1) && (d4<nrad) &&
             (abs(result.surface[i][ku]-d1)<2) && 
             (abs(result.surface[i+1][ku]-d2)<2) &&
             (abs(result.surface[i+2][ku]-d3)<2) && 
             (abs(result.surface[i+3][ku]-d4)<2) &&
             (alpha[m][jj] > tscore))
            {
              tscore = alpha[m][jj];
              a1 = m;

              result.surface[i][jj+kl] = d1;
              result.surface[i+1][jj+kl] = d2;
              result.surface[i+2][jj+kl] = d3;
              result.surface[i+3][jj+kl] = d4;
              for(j=jj;j>0;j--)
                {
                  a1 = bptr[a1][j];
                  d1 = a1/27;
                  d1=result.surface[i-1][kl+j-1] + d1-1;
                  d2 = (a1%27)/9;
                  d2 = d1 + d2 -1;
                  d3 = (a1%9)/3;
                  d3 = d2 + d3 -1;
                  d4 = a1%3;
                  d4 = d3 + d4 -1;

                  result.surface[i][kl+j-1] = d1;
                  result.surface[i+1][kl+j-1] = d2;
                  result.surface[i+2][kl+j-1] = d3;
                  result.surface[i+3][kl+j-1] = d4;
                }
            }
        }
    
      if(tscore == 0)
        {
          printf("ERROR: Can not compute a quad constant layer i=%d Hem = %d\n"
                 ,i,ihm);
          return(0);
        }
    }
  return(tscore);
}

int vtkCellWallVisSeg::ddynam0(int i, int knum)
{
  
  register int j,k;
  int ihm,kl,kll,ku,ks;
  int nrad, tscore;
  int alpha [9][MAX_COLS];
  int bptr [9][MAX_COLS];
  int a1,a3;
  int b1, b2, b3, b4, m;
  int c1, c2, kk;

  /* get some parameters */
  nrad = image.nrad;

  if((image.conn[i-1] !=0) || (image.conn[i] != -1))
    {
      printf("ERROR:  Called ddynam0 with incorrect conditions %d %d\n",
             image.conn[i-1], image.conn[i]);
      exit(9);
    }


  for(ihm=0; ihm<2; ihm++)
    {
      kl = (knum/2)*ihm;
      kll = kl/2;
      ku = (knum/2)*(1-ihm);

      tscore = 0;

      for(k=0;k<9;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;
              
      c1 = result.surface[i][kl];
      c2 = result.surface[i+1][kll];
      kk = 3*(c1 - result.surface[i-1][kl]+1) + (c2-c1+1);

      alpha[kk][0] = image2.data[i][kl][c1] + image2.data[i+1][kll][c2];
              

      for(j=1;j<(knum/2);j++)
        for(k=0; k<9; k++)
          if(alpha[k][j-1] != 0)
            {
              b1 = k/3;
              b1 = result.surface[i-1][kl+j-1] + b1 -1;
              b2 = k%3;
              b2 = b1+b2-1;
              if((b1>-1) && (b1<nrad) && (b2>-1) && (b2<nrad))
                for(m=0;m<9;m++)
                  {
                    b3=m/3;
                    b3=result.surface[i-1][kl+j] + b3-1;
                    b4 = m%3;
                    b4 = b3+b4-1;
                    if((b3>-1) && (b3<nrad) && 
                       (b4>-1) && (b4<nrad) &&
                       (abs(b3-b1)<2))
                      {
                        a3=0;
                        if(((j%2) == 1) && (b2 == b4))
                          a3 = alpha[k][j-1] +
                            image2.data[i][kl+j][b3];
                        if(((j%2) == 0) && (abs(b2-b4) < 2))
                          a3 = alpha[k][j-1] +
                            image2.data[i][kl+j][b3] +
                            image2.data[i+1][kll+j/2][b4];

                        if(alpha[m][j] < a3)
                          {
                            alpha[m][j] = a3;
                            bptr[m][j] = k;
                          }
                      }
                  }
            }
                  
      for(m=0;m<9;m++)
        {
          b3=m/3;
          ks = kl + (knum/2) -1;
          b3=result.surface[i-1][ks] + b3-1;
          b4 = m%3;
          b4 = b3+b4-1;
          if((b3>-1) && (b3<nrad) &&
             (b4>-1) && (b4<nrad) &&
             (abs(b3-result.surface[i][ku])<2) &&
             (abs(b4-result.surface[i+1][ku/2])<2) &&
             (alpha[m][knum/2-1] > tscore))
            {
              tscore = alpha[m][knum/2-1];
              a1 = m;
              result.surface[i][ks] = b3;
              result.surface[i+1][ks/2] = b4;
              for(j=(knum/2-1);j>1;j--)
                {
                  a1 = bptr[a1][j];
                  b3=a1/3;
                  b3=result.surface[i-1][kl+j-1] + b3-1;
                  b4 = a1%3;
                  b4 = b3+b4-1;
                  result.surface[i][kl+j-1] = b3;
                  result.surface[i+1][(kl+j-1)/2] = b4;
                }
            }
        }
 
      if(tscore == 0)
        {
          printf("ERROR: Can not compute a reduction layer. i=%d on hemisphere %d \n",i,ihm);

          for(j=0;j<knum;j++)
            {
              printf("%d ",result.surface[i-1][j]);
              if(j%16 == 15) printf("\n");
            }
          k = knum/2;
          printf("%d %d %d\n",0,result.surface[i][0],
                   result.surface[i][k]);
          printf("%d %d %d\n",1,result.surface[i+1][0],
                 result.surface[i+1][k/2]);

          return(0);
        }
    }

  return(tscore);
}

int vtkCellWallVisSeg::ddynam2(int i, int knum)
{
  
  register int j,k;
  int ihm,kl,kll,ku,ks;
  int nrad, tscore;
  int alpha [9][MAX_COLS];
  int bptr [9][MAX_COLS];
  int a1,a3;
  int b1, b2, b3, b4, m;
  int c1, c2, kk;

  /* get some parameters */
  nrad = image.nrad;

  if((image.conn[i-1] !=0) || (image.conn[i] != 0))
    {
      printf("ERROR:  Called ddynam2 with incorrect conditions %d %d\n",
             image.conn[i-1], image.conn[i]);
      exit(9);
    }

  for(ihm=0; ihm<2; ihm++)
    {
      kl = (knum/2)*ihm;
      kll = kl/2;
      ku = (knum/2)*(1-ihm);

      tscore = 0;

      for(k=0;k<9;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;
              
      c1 = result.surface[i][kl];
      c2 = result.surface[i+1][kl];
      kk = 3*(c1 - result.surface[i-1][kl]+1) + (c2-c1+1);

      alpha[kk][0] = image2.data[i][kl][c1] + image2.data[i+1][kl][c2];
              
      for(j=1;j<(knum/2);j++)
        for(k=0; k<9; k++)
          if(alpha[k][j-1] != 0)
            {
              b1 = k/3;
              b1 = result.surface[i-1][kl+j-1] + b1 -1;
              b2 = k%3;
              b2 = b1+b2-1;
              if((b1>-1) && (b1<nrad) && (b2>-1) && (b2<nrad))
                for(m=0;m<9;m++)
                  {
                    b3=m/3;
                    b3=result.surface[i-1][kl+j] + b3-1;
                    b4 = m%3;
                    b4 = b3+b4-1;
                    if((b3>-1) && (b3<nrad) && 
                       (b4>-1) && (b4<nrad) &&
                       (abs(b3-b1)<2) && (abs(b4-b2)<2))
                      {
                        a3 = alpha[k][j-1] +
                          image2.data[i][kl+j][b3] +
                          image2.data[i+1][kl+j][b4];
                            
                        if(alpha[m][j] < a3)
                          {
                            alpha[m][j] = a3;
                            bptr[m][j] = k;
                          }
                      }
                  }
            }
                  
      for(m=0;m<9;m++)
        {
          b3=m/3;
          ks = kl + (knum/2) -1;
          b3=result.surface[i-1][ks] + b3-1;
          b4 = m%3;
          b4 = b3+b4-1;
          if((b3>-1) && (b3<nrad) &&
             (b4>-1) && (b4<nrad) &&
             (abs(b3-result.surface[i][ku])<2) &&
             (abs(b4-result.surface[i+1][ku])<2) &&
             (alpha[m][knum/2-1] > tscore))
            {
              tscore = alpha[m][knum/2-1];
              a1 = m;
              result.surface[i][ks] = b3;
              result.surface[i+1][ks] = b4;
              for(j=(knum/2-1);j>0;j--)
                {
                  a1 = bptr[a1][j];
                  b3=a1/3;
                  b3=result.surface[i-1][kl+j-1] + b3-1;
                  b4 = a1%3;
                  b4 = b3+b4-1;
                  result.surface[i][kl+j-1] = b3;
                  result.surface[i+1][kl+j-1] = b4;
                }
            }
        }
 
      if(tscore == 0)
        {
          printf("ERROR: Can not compute a double constant layer. i=%d on hemisphere %d \n",i,ihm);
          return(0);
        }
    }

  return(tscore);
}

int vtkCellWallVisSeg::ddynam3(int i, int knum)
{
  
  register int j,k;
  int ihm,kl,kll,ku,ks;
  int nrad, tscore;
  int alpha [9][MAX_COLS];
  int bptr [9][MAX_COLS];
  int a1,a3;
  int b1, b2, b3, b4, m;
  int c1, c2, kk;

  /* get some parameters */
  nrad = image.nrad;

  if((image.conn[i-1] !=0) || (image.conn[i] != 1))
    {
      printf("ERROR:  Called ddynam3 with incorrect conditions %d %d\n",
             image.conn[i-1], image.conn[i]);
      exit(9);
    }

  for(ihm=0; ihm<2; ihm++)
    {
      kl = (knum/2)*ihm;
      kll = knum*ihm;
      ku = (knum/2)*(1-ihm);

      tscore = 0;

      for(k=0;k<9;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;
              
      c1 = result.surface[i][kl];
      c2 = result.surface[i+1][kll];
      kk = 3*(c1 - result.surface[i-1][kl]+1) + (c2-c1+1);

      alpha[kk][0] = image2.data[i][kl][c1] + image2.data[i+1][kll][c2];
              
      for(j=1;j<knum;j++)
        for(k=0; k<9; k++)
          if(alpha[k][j-1] != 0)
            {
              b1 = k/3;
              b1 = result.surface[i-1][kl+(j-1)/2] + b1 -1;
              b2 = k%3;
              b2 = b1+b2-1;

              if((b1>-1) && (b1<nrad) && (b2>-1) && (b2<nrad))
                for(m=0;m<9;m++)
                  {
                    b3=m/3;
                    b3=result.surface[i-1][kl+j/2] + b3-1;
                    b4 = m%3;
                    b4 = b3+b4-1;
                    if((b3>-1) && (b3<nrad) && 
                       (b4>-1) && (b4<nrad) &&
                       (abs(b4-b2)<2))
                      {
                        a3 = 0;
                        if(((j%2) ==1) && (b3 == b1))
                          a3 = alpha[k][j-1] +
                            image2.data[i+1][j+kll][b4];
                        if(((j%2) ==0) && (abs(b3-b1)<2))
                          a3 = alpha[k][j-1] +
                            image2.data[i+1][j+kll][b4] +
                            image2.data[i][kl+j/2][b3];
                            
                        if(alpha[m][j] < a3)
                          {
                            alpha[m][j] = a3;
                            bptr[m][j] = k;
                          }
                      }
                  }
            }
                  
      for(m=0;m<9;m++)
        {
          b3=m/3;
          ks = kl + (knum/2) -1;
          b3=result.surface[i-1][ks] + b3-1;
          b4 = m%3;
          b4 = b3+b4-1;

          if((b3>-1) && (b3<nrad) &&
             (b4>-1) && (b4<nrad) &&
             (abs(b3-result.surface[i][ku])<2) &&
             (abs(b4-result.surface[i+1][2*ku])<2) &&
             (alpha[m][knum-1] > tscore))
            {
              tscore = alpha[m][knum-1];
              a1 = m;
              result.surface[i][ks] = b3;
              result.surface[i+1][ks*2+1] = b4;
              for(j=(knum-1);j>0;j--)
                {
                  a1 = bptr[a1][j];
                  b3=a1/3;
                  b3=result.surface[i-1][kl+(j-1)/2] + b3-1;
                  b4 = a1%3;
                  b4 = b3+b4-1;
                  result.surface[i][kl+(j-1)/2] = b3;
                  result.surface[i+1][2*kl+j-1] = b4;
                }
            }
        }
 
      if(tscore == 0)
        {
          printf("ERROR: Can not compute a expansion layer. i=%d on hemisphere %d \n",i,ihm);
          return(0);
        }
    }

  return(tscore);
}

int vtkCellWallVisSeg::ddynam4(int i, int knum)
{
  
  register int j,k;
  int ihm,kl,kll,ku,ks;
  int nrad, tscore;
  int alpha [9][MAX_COLS];
  int bptr [9][MAX_COLS];
  int a1,a3;
  int b1, b2, b3, b4, m;
  int c1, c2, kk;

  /* get some parameters */
  nrad = image.nrad;

  if((image.conn[i-1] !=1) || (image.conn[i] != 0))
    {
      printf("ERROR:  Called ddynam4 with incorrect conditions %d %d\n",
             image.conn[i-1], image.conn[i]);
      exit(9);
    }

  for(ihm=0; ihm<2; ihm++)
    {
      kl = (knum/2)*ihm;
      kll = kl/2;
      ku = (knum/2)*(1-ihm);

      tscore = 0;

      for(k=0;k<9;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;
              
      c1 = result.surface[i][kl];
      c2 = result.surface[i+1][kl];
      kk = 3*(c1 - result.surface[i-1][kll]+1) + (c2-c1+1);

      alpha[kk][0] = image2.data[i][kl][c1] + image2.data[i+1][kl][c2];
              
      for(j=1;j<(knum/2);j++)
        for(k=0; k<9; k++)
          if(alpha[k][j-1] != 0)
            {
              b1 = k/3;
              b1 = result.surface[i-1][(kl+j-1)/2] + b1 -1;
              b2 = k%3;
              b2 = b1+b2-1;
              if((b1>-1) && (b1<nrad) && (b2>-1) && (b2<nrad))
                for(m=0;m<9;m++)
                  {
                    b3=m/3;
                    b3=result.surface[i-1][(kl+j)/2] + b3-1;
                    b4 = m%3;
                    b4 = b3+b4-1;
                    if((b3>-1) && (b3<nrad) && 
                       (b4>-1) && (b4<nrad) &&
                       (abs(b3-b1)<2) && (abs(b4-b2)<2))
                      {
                        a3 = alpha[k][j-1] +
                          image2.data[i][kl+j][b3] +
                          image2.data[i+1][kl+j][b4];
                            
                        if(alpha[m][j] < a3)
                          {
                            alpha[m][j] = a3;
                            bptr[m][j] = k;

                          }
                      }
                  }
            }
                  
      for(m=0;m<9;m++)
        {
          b3=m/3;
          ks = kl + (knum/2) -1;
          b3=result.surface[i-1][ks/2] + b3-1;
          b4 = m%3;
          b4 = b3+b4-1;

          if((b3>-1) && (b3<nrad) &&
             (b4>-1) && (b4<nrad) &&
             (abs(b3-result.surface[i][ku])<2) &&
             (abs(b4-result.surface[i+1][ku])<2) &&
             (alpha[m][knum/2-1] > tscore))
            {
              tscore = alpha[m][knum/2-1];
              a1 = m;
              result.surface[i][ks] = b3;
              result.surface[i+1][ks] = b4;
              for(j=(knum/2-1);j>0;j--)
                {
                  a1 = bptr[a1][j];
                  b3=a1/3;
                  b3=result.surface[i-1][(kl+j-1)/2] + b3-1;
                  b4 = a1%3;
                  b4 = b3+b4-1;
                  result.surface[i][kl+j-1] = b3;
                  result.surface[i+1][kl+j-1] = b4;
                }
            }
        }
 
      if(tscore == 0)
        {
          printf("ERROR: Can not compute a post expansion layer. i=%d on hemisphere %d \n",i,ihm);
          return(0);
        }
    }

  return(tscore);
}

int vtkCellWallVisSeg::qdynams2(int i, int knum)
{
  register int j,k;
  int ihm,kk,kl,ku,jj;
  int nrad, tscore;
  int alpha [81][MAX_COLS];
  int bptr [81][MAX_COLS];
  int a1,a3;
  int b1, b2, b3, b4, m, m1, m2, m3, m4;
  int c1, c2, c3, c4;
  int d1, d2, d3, d4;


  /* get some parameters */
  nrad = image.nrad;
  if((image.conn[i] !=0) || (image.conn[i-1] != 0) ||
     (image.conn[i-2] !=0) || (image.conn[i-3] != 0) )
    {
      printf("ERROR:  Called qdynams2 with incorrect conditions %d %d %d %d\n",
             image.conn[i], image.conn[i-1], image.conn[i-2], image.conn[i-3]);
      exit(9);
    }

  for(ihm=0;ihm<2;ihm++)
    {
      kl = (knum/2)*ihm;
      ku = (1-ihm)*(knum/2);

      tscore = 0;

      for(k=0;k<81;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;
              
      c1 = result.surface[i][kl];
      c2 =  result.surface[i-1][kl];
      c3 =  result.surface[i-2][kl];
      c4 =  result.surface[i-3][kl];

      kk = 27*(c1-result.surface[i+1][kl]+1) + 9*(c2-c1+1) + 3*(c3-c2+1) +
        (c4-c3+1);

      alpha[kk][0] = image2.data[i][kl][c1] + image2.data[i-1][kl][c2] +
            image2.data[i-2][kl][c3] + image2.data[i-3][kl][c4];
              
      for(j=1;j<(knum/2);j++)
        for(k=0; k<81; k++)
          if(alpha[k][j-1] != 0)
            {
              b1 = k/27;
              b1 = result.surface[i+1][j+kl-1] + b1 -1;
              b2 = (k%27)/9;
              b2 = b1+b2-1;
              b3 = (k%9)/3;
              b3 = b2 + b3 -1;
              b4 = k%3;
              b4 = b3 + b4 -1;
              if((b1>-1) && (b1<nrad) && (b2>-1) && (b2<nrad) &&
                 (b3>-1) && (b3<nrad) && (b4>-1) && (b4<nrad))
                for(m1=-1;m1<2;m1++)
                  {
                    d1=result.surface[i+1][j+kl] + m1;
                    if((d1>-1) && (d1<nrad) && (abs(b1-d1)<2))
                      {
                        for(m2 =-1;m2<2;m2++)
                          {
                            d2 = d1+m2;
                            if((d2>-1) && (d2<nrad) && (abs(b2-d2)<2))
                              {
                                for(m3 =-1;m3<2;m3++)
                                  {
                                    d3 = d2 + m3;
                                    if((d3>-1) && (d3<nrad) && (abs(b3-d3)<2))
                                      {
                                        for(m4 =-1;m4<2;m4++)
                                          {
                                            d4 = d3 + m4;
                                            if((d4>-1) && (d4<nrad) && 
                                               (abs(b4-d4)<2))
                                              {
                                                m = 27*(m1+1) + 9*(m2+1) 
                                                  + 3*(m3+1) +m4+1;
                        a3 = alpha[k][j-1] +
                          image2.data[i][j+kl][d1] +
                          image2.data[i-1][j+kl][d2] +
                          image2.data[i-2][j+kl][d3] +
                          image2.data[i-3][j+kl][d4];
                            
                        if(alpha[m][j] < a3)
                          {
                            alpha[m][j] = a3;
                            bptr[m][j] = k;
                          }
                      }
                  }
            }
                                  }}}}}}

      jj=(knum/2)-1;

      for(m=0;m<81;m++)
        {
          d1 = m/27;
          d1=result.surface[i+1][kl + jj] + d1-1;
          d2 = (m%27)/9;
          d2 = d1 + d2 -1;
          d3 = (m%9)/3;
          d3 = d2 + d3 -1;
          d4 = m%3;
          d4 = d3 + d4 -1;

          if((d1>-1) && (d1<nrad) &&
             (d2>-1) && (d2<nrad) &&
             (d3>-1) && (d3<nrad) &&
             (d4>-1) && (d4<nrad) &&
             (abs(result.surface[i][ku]-d1)<2) && 
             (abs(result.surface[i-1][ku]-d2)<2) &&
             (abs(result.surface[i-2][ku]-d3)<2) && 
             (abs(result.surface[i-3][ku]-d4)<2) &&
             (alpha[m][jj] > tscore))
            {
              tscore = alpha[m][jj];
              a1 = m;

              result.surface[i][jj+kl] = d1;
              result.surface[i-1][jj+kl] = d2;
              result.surface[i-2][jj+kl] = d3;
              result.surface[i-3][jj+kl] = d4;
              for(j=jj;j>0;j--)
                {
                  a1 = bptr[a1][j];
                  d1 = a1/27;
                  d1=result.surface[i+1][kl+j-1] + d1-1;
                  d2 = (a1%27)/9;
                  d2 = d1 + d2 -1;
                  d3 = (a1%9)/3;
                  d3 = d2 + d3 -1;
                  d4 = a1%3;
                  d4 = d3 + d4 -1;

                  result.surface[i][kl+j-1] = d1;
                  result.surface[i-1][kl+j-1] = d2;
                  result.surface[i-2][kl+j-1] = d3;
                  result.surface[i-3][kl+j-1] = d4;
                }
            }
        }
    
      if(tscore == 0)
        {
          printf("ERROR: Can not compute a quad constant layer i=%d Hem = %d\n"
                 ,i,ihm);
          return(0);
        }
    }

  return(tscore);
}

int vtkCellWallVisSeg::ddynams2(int i, int knum)
{
  
  register int j,k;
  int ihm,kl,kll,ku,ks;
  int nrad, tscore;
  int alpha [9][MAX_COLS];
  int bptr [9][MAX_COLS];
  int a1,a3;
  int b1, b2, b3, b4, m;
  int c1, c2, kk;

  /* get some parameters */
  nrad = image.nrad;

  if((image.conn[i] !=0) || (image.conn[i-1] != 0))
    {
      printf("ERROR:  Called ddynams2 with incorrect conditions %d %d\n",
             image.conn[i], image.conn[i-1]);
      exit(9);
    }

  for(ihm=0; ihm<2; ihm++)
    {
      kl = (knum/2)*ihm;
      kll = kl/2;
      ku = (knum/2)*(1-ihm);

      tscore = 0;

      for(k=0;k<9;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;
              
      c1 = result.surface[i][kl];
      c2 = result.surface[i-1][kl];
      kk = 3*(c1 - result.surface[i+1][kl]+1) + (c2-c1+1);

      alpha[kk][0] = image2.data[i][kl][c1] + image2.data[i-1][kl][c2];
              
      for(j=1;j<(knum/2);j++)
        for(k=0; k<9; k++)
          if(alpha[k][j-1] != 0)
            {
              b1 = k/3;
              b1 = result.surface[i+1][kl+j-1] + b1 -1;
              b2 = k%3;
              b2 = b1+b2-1;
              if((b1>-1) && (b1<nrad) && (b2>-1) && (b2<nrad))
                for(m=0;m<9;m++)
                  {
                    b3=m/3;
                    b3=result.surface[i+1][kl+j] + b3-1;
                    b4 = m%3;
                    b4 = b3+b4-1;
                    if((b3>-1) && (b3<nrad) && 
                       (b4>-1) && (b4<nrad) &&
                       (abs(b3-b1)<2) && (abs(b4-b2)<2))
                      {
                        a3 = alpha[k][j-1] +
                          image2.data[i][kl+j][b3] +
                          image2.data[i-1][kl+j][b4];
                            
                        if(alpha[m][j] < a3)
                          {
                            alpha[m][j] = a3;
                            bptr[m][j] = k;
                          }
                      }
                  }
            }
                  
      for(m=0;m<9;m++)
        {
          b3=m/3;
          ks = kl + (knum/2) -1;
          b3=result.surface[i+1][ks] + b3-1;
          b4 = m%3;
          b4 = b3+b4-1;
          if((b3>-1) && (b3<nrad) &&
             (b4>-1) && (b4<nrad) &&
             (abs(b3-result.surface[i][ku])<2) &&
             (abs(b4-result.surface[i-1][ku])<2) &&
             (alpha[m][knum/2-1] > tscore))
            {
              tscore = alpha[m][knum/2-1];
              a1 = m;
              result.surface[i][ks] = b3;
              result.surface[i-1][ks] = b4;
              for(j=(knum/2-1);j>0;j--)
                {
                  a1 = bptr[a1][j];
                  b3=a1/3;
                  b3=result.surface[i+1][kl+j-1] + b3-1;
                  b4 = a1%3;
                  b4 = b3+b4-1;
                  result.surface[i][kl+j-1] = b3;
                  result.surface[i-1][kl+j-1] = b4;
                }
            }
        }
 
      if(tscore == 0)
        {
          printf("ERROR: Can not compute a south double constant layer. i=%d on hemisphere %d \n",i,ihm);
          return(0);
        }
    }

  return(tscore);
}

int vtkCellWallVisSeg::ddynams3(int i, int knum)
{
  
  register int j,k;
  int ihm,kl,kll,ku,ks;
  int nrad, tscore;
  int alpha [9][MAX_COLS];
  int bptr [9][MAX_COLS];
  int a1,a3;
  int b1, b2, b3, b4, m;
  int c1, c2, kk;

  /* get some parameters */
  nrad = image.nrad;

  if((image.conn[i] !=0) || (image.conn[i-1] != -1))
    {
      printf("ERROR:  Called ddynams3 with incorrect conditions %d %d\n",
             image.conn[i], image.conn[i-1]);
      exit(9);
    }

  for(ihm=0; ihm<2; ihm++)
    {
      kl = (knum/2)*ihm;
      kll = knum*ihm;
      ku = (knum/2)*(1-ihm);

      tscore = 0;

      for(k=0;k<9;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;
              
      c1 = result.surface[i][kl];
      c2 = result.surface[i-1][kll];
      kk = 3*(c1 - result.surface[i+1][kl]+1) + (c2-c1+1);

      alpha[kk][0] = image2.data[i][kl][c1] + image2.data[i-1][kll][c2];
              
      for(j=1;j<knum;j++)
        for(k=0; k<9; k++)
          if(alpha[k][j-1] != 0)
            {
              b1 = k/3;
              b1 = result.surface[i+1][kl+(j-1)/2] + b1 -1;
              b2 = k%3;
              b2 = b1+b2-1;

              if((b1>-1) && (b1<nrad) && (b2>-1) && (b2<nrad))
                for(m=0;m<9;m++)
                  {
                    b3=m/3;
                    b3=result.surface[i+1][kl+j/2] + b3-1;
                    b4 = m%3;
                    b4 = b3+b4-1;
                    if((b3>-1) && (b3<nrad) && 
                       (b4>-1) && (b4<nrad) &&
                       (abs(b4-b2)<2))
                      {
                        a3 = 0;
                        if(((j%2) ==1) && (b3 == b1))
                          a3 = alpha[k][j-1] +
                            image2.data[i-1][j+kll][b4];
                        if(((j%2) ==0) && (abs(b3-b1)<2))
                          a3 = alpha[k][j-1] +
                            image2.data[i-1][j+kll][b4] +
                            image2.data[i][kl+j/2][b3];
                            
                        if(alpha[m][j] < a3)
                          {
                            alpha[m][j] = a3;
                            bptr[m][j] = k;
                          }
                      }
                  }
            }
                  
      for(m=0;m<9;m++)
        {
          b3=m/3;
          ks = kl + (knum/2) -1;
          b3=result.surface[i+1][ks] + b3-1;
          b4 = m%3;
          b4 = b3+b4-1;

          if((b3>-1) && (b3<nrad) &&
             (b4>-1) && (b4<nrad) &&
             (abs(b3-result.surface[i][ku])<2) &&
             (abs(b4-result.surface[i-1][2*ku])<2) &&
             (alpha[m][knum-1] > tscore))
            {
              tscore = alpha[m][knum-1];
              a1 = m;
              result.surface[i][ks] = b3;
              result.surface[i-1][ks*2+1] = b4;
              for(j=(knum-1);j>0;j--)
                {
                  a1 = bptr[a1][j];
                  b3=a1/3;
                  b3=result.surface[i+1][kl+(j-1)/2] + b3-1;
                  b4 = a1%3;
                  b4 = b3+b4-1;
                  result.surface[i][kl+(j-1)/2] = b3;
                  result.surface[i-1][2*kl+j-1] = b4;
                }
            }
        }
 
      if(tscore == 0)
        {
          printf("ERROR: Can not compute a south expansion layer. i=%d on hemisphere %d \n",i,ihm);
          return(0);
        }
    }

  return(tscore);
}

int vtkCellWallVisSeg::ddynams4(int i, int knum)
{
  
  register int j,k;
  int ihm,kl,kll,ku,ks;
  int nrad, tscore;
  int alpha [9][MAX_COLS];
  int bptr [9][MAX_COLS];
  int a1,a3;
  int b1, b2, b3, b4, m;
  int c1, c2, kk;

  /* get some parameters */
  nrad = image.nrad;

  if((image.conn[i] != -1) || (image.conn[i-1] != 0))
    {
      printf("ERROR:  Called ddynams4 with incorrect conditions %d %d\n",
             image.conn[i], image.conn[i-1]);
      exit(9);
    }

  for(ihm=0; ihm<2; ihm++)
    {
      kl = (knum/2)*ihm;
      kll = kl/2;
      ku = (knum/2)*(1-ihm);

      tscore = 0;

      for(k=0;k<9;k++)
        for(j=0;j<knum;j++)
          alpha[k][j]=0;
              
      c1 = result.surface[i][kl];
      c2 = result.surface[i-1][kl];
      kk = 3*(c1 - result.surface[i+1][kll]+1) + (c2-c1+1);

      alpha[kk][0] = image2.data[i][kl][c1] + image2.data[i-1][kl][c2];
              
      for(j=1;j<(knum/2);j++)
        for(k=0; k<9; k++)
          if(alpha[k][j-1] != 0)
            {
              b1 = k/3;
              b1 = result.surface[i+1][(kl+j-1)/2] + b1 -1;
              b2 = k%3;
              b2 = b1+b2-1;
              if((b1>-1) && (b1<nrad) && (b2>-1) && (b2<nrad))
                for(m=0;m<9;m++)
                  {
                    b3=m/3;
                    b3=result.surface[i+1][(kl+j)/2] + b3-1;
                    b4 = m%3;
                    b4 = b3+b4-1;
                    if((b3>-1) && (b3<nrad) && 
                       (b4>-1) && (b4<nrad) &&
                       (abs(b3-b1)<2) && (abs(b4-b2)<2))
                      {
                        a3 = alpha[k][j-1] +
                          image2.data[i][kl+j][b3] +
                          image2.data[i-1][kl+j][b4];
                            
                        if(alpha[m][j] < a3)
                          {
                            alpha[m][j] = a3;
                            bptr[m][j] = k;

                          }
                      }
                  }
            }
      
      for(m=0;m<9;m++)
        {
          b3=m/3;
          ks = kl + (knum/2) -1;
          b3=result.surface[i+1][ks/2] + b3-1;
          b4 = m%3;
          b4 = b3+b4-1;

          if((b3>-1) && (b3<nrad) &&
             (b4>-1) && (b4<nrad) &&
             (abs(b3-result.surface[i][ku])<2) &&
             (abs(b4-result.surface[i-1][ku])<2) &&
             (alpha[m][knum/2-1] > tscore))
            {
              tscore = alpha[m][knum/2-1];
              a1 = m;
              result.surface[i][ks] = b3;
              result.surface[i-1][ks] = b4;
              for(j=(knum/2-1);j>0;j--)
                {
                  a1 = bptr[a1][j];
                  b3=a1/3;
                  b3=result.surface[i+1][(kl+j-1)/2] + b3-1;
                  b4 = a1%3;
                  b4 = b3+b4-1;
                  result.surface[i][kl+j-1] = b3;
                  result.surface[i-1][kl+j-1] = b4;
                }
            }
        }
 
      if(tscore == 0)
        {
          printf("ERROR: Can not compute a south post expansion layer. i=%d on hemisphere %d \n",i,ihm);
          return(0);
        }
    }

  return(tscore);
}

