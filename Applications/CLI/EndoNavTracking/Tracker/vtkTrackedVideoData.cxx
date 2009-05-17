/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTrackedVideoData.cxx,v $
  Language:  C++
  Date:      $Date: 2008/12/29 18:39:45 $
  Version:   $Revision: 1.15 $

  =========================================================================*/
#include "vtkTrackedVideoData.h"
#include "vtkObjectFactory.h"
#include <iostream>

using namespace std;

vtkCxxRevisionMacro(vtkTrackedVideoData, "$Revision: 1.15 $");
vtkStandardNewMacro(vtkTrackedVideoData);

vtkTrackedVideoData::vtkTrackedVideoData() {

  this->ImageStored = 0;
  this->NumPreallocFrames = 0;
  this->NumPreallocFramesRecorded = 0;
  this->Preallocated = 0;

  this->Times            = vtkDoubleArray::New();
  this->Times->SetNumberOfComponents(16);
  this->RegMatrix          = vtkMatrix4x4::New();
  this->CalibMatrix        = vtkMatrix4x4::New();
  this->SensorMatrixCollection  = vtkCollection::New();
  this->ImageCollection      = vtkCollection::New();

  this->prealloc_out        = vtkImageData::New();

  this->DataPositions = vtkPoints::New();
  this->DataPositionsAfterRegistration = vtkPoints::New();
  this->DataElementAsArray = vtkDoubleArray::New();

  this->Reset();
}

vtkTrackedVideoData::~vtkTrackedVideoData() {
  if (this->RegMatrix != NULL) {
    this->RegMatrix->Delete();
  }
  if (this->CalibMatrix != NULL) {
    this->CalibMatrix->Delete();
  }
  if (this->Times != NULL) {
    this->Times->Delete();
  }
  if (this->DataElementAsArray != NULL) {
    this->DataElementAsArray->Delete();
  }
  if (this->SensorMatrixCollection != NULL) {
    this->SensorMatrixCollection->Delete();
  }
  if (this->ImageCollection != NULL) {
    this->ImageCollection->Delete();
  }
  if (this->prealloc_out != NULL) {
    this->prealloc_out->Delete();
  }
  if (this->DataPositions != NULL) {
    this->DataPositions->Delete();
  }
  if (this->DataPositionsAfterRegistration != NULL) {
    this->DataPositionsAfterRegistration->Delete();
  }

}


void vtkTrackedVideoData::PrintSelf(ostream &os, vtkIndent indent) {

  //SuperClass::PrintSelf(os,indent);
}


void vtkTrackedVideoData::Reset(void) {
  
  int n;

  this->Times->Initialize();

  for (n=this->SensorMatrixCollection->GetNumberOfItems(); n>0; n--) {
    this->SensorMatrixCollection->RemoveItem(n-1);
  }

  for (n=this->ImageCollection->GetNumberOfItems(); n>0; n--) {
    this->ImageCollection->RemoveItem(n-1);
  }
  this->ImageStored = 0;

  this->Preallocate(0);
}


void vtkTrackedVideoData::Preallocate(int numFrames) {

  if (numFrames > 0 && !this->ImageStored) {
  if (this->prealloc_out != NULL)
  {
      this->prealloc_out->Delete();
  }
    this->prealloc_out = vtkImageData::New();
    this->NumPreallocFrames = numFrames;
    this->NumPreallocFramesRecorded = 0;
  } else {
  if (this->prealloc_out != NULL)
  {
      this->prealloc_out->Delete();
  }
    this->prealloc_out = vtkImageData::New();
    this->NumPreallocFrames = 0;
    this->NumPreallocFramesRecorded = 0;
  }

  this->Preallocated = 0;
}


/****************************************
*  Routines for Collecting Data
*
*/
void vtkTrackedVideoData::SetRegistration(vtkMatrix4x4 *Matrix) {

  this->RegMatrix->DeepCopy(Matrix);
}


void vtkTrackedVideoData::SetCalibration(vtkMatrix4x4 *Matrix) {
  this->CalibMatrix->DeepCopy(Matrix);
}


void vtkTrackedVideoData::AddDataElement(double timeStamp, vtkMatrix4x4 *SensorMatrix) {

  vtkMatrix4x4 *temp = vtkMatrix4x4::New();

  temp->DeepCopy(SensorMatrix);
  this->ImageStored = 0;
  this->SensorMatrixCollection->AddItem(temp);
  this->Times->InsertComponent(this->Times->GetNumberOfTuples(),0,timeStamp);
  for (int n=1; n<16; n++)
    this->Times->InsertComponent(this->Times->GetNumberOfTuples(),n,0.0);

  // Delete the object to avoid having two reference count (one by the collection and
  // one by the object itself
  temp->Delete();

}


void vtkTrackedVideoData::AddDataElementV(double timeStamp, vtkMatrix4x4 *SensorMatrix, vtkImageData *image) {

  // images
  // check data - can't collect sensor data sometimes and images other times
  if (this->SensorMatrixCollection->GetNumberOfItems() == this->ImageCollection->GetNumberOfItems() ||
    this->SensorMatrixCollection->GetNumberOfItems() == this->NumPreallocFramesRecorded) {
    // data is in sync - record images
    this->ImageStored = 1;    
    if (this->NumPreallocFrames == 0) {
      // use collection (memory intensive, flexible)
      vtkImageData *itemp = vtkImageData::New();
      image->UpdateInformation();
      image->UpdateData();
      itemp->DeepCopy(image);
      this->ImageCollection->AddItem(itemp);
    itemp->Delete();
    } else {
      // use preallocated buffer (memory light, inflexible)
      this->preallocInsertSlice(image,this->NumPreallocFramesRecorded++);
    }
  } else {
    // data is not in sync - don't record images
    this->ImageStored = 0;
  }

  // sensor data
  vtkMatrix4x4 *temp = vtkMatrix4x4::New();
  temp->DeepCopy(SensorMatrix);
  this->SensorMatrixCollection->AddItem(temp);
  
  // time stamps
  this->Times->InsertComponent(this->Times->GetNumberOfTuples(),0,timeStamp);
  for (int n=1; n<16; n++)
    this->Times->InsertComponent(this->Times->GetNumberOfTuples(),n,0.0);

  // Remove created objects to avoid having 2 reference count
  temp->Delete();
}


void vtkTrackedVideoData::preallocInsertSlice(vtkImageData *inData, int slice) {
  int idxR, idxY;
  int inIncX, inIncY, inIncZ;
  int outIncX, outIncY, outIncZ;
  int rowLength, Ext[6];

  int *dims = inData->GetDimensions();

  rowLength = dims[0]*inData->GetNumberOfScalarComponents();
  Ext[0] = 0;
  Ext[1] = dims[0]-1;
  Ext[2] = 0;
  Ext[3] = dims[1]-1;
  Ext[4] = 0;
  Ext[5] = 1;

  // preallocate output buffer if not already done
  if (!this->Preallocated) {
    this->prealloc_out->SetScalarTypeToUnsignedChar();
    this->prealloc_out->SetDimensions(dims[0],dims[1],this->NumPreallocFrames);
    this->prealloc_out->SetNumberOfScalarComponents(inData->GetNumberOfScalarComponents());
    try {
      this->prealloc_out->AllocateScalars();
      this->Preallocated = 1;
    } catch(char * str) {
      vtkErrorMacro(<< "Exception: " << str);
      return;
    }
  }

  unsigned char *inPtr  = (unsigned char *) inData->GetScalarPointer();
  unsigned char *outPtr = (unsigned char *) this->prealloc_out->GetScalarPointer(0,0,slice);

  // Get increments to march through data 
  inData->GetContinuousIncrements(Ext, inIncX, inIncY, inIncZ);
  this->prealloc_out->GetContinuousIncrements(Ext, outIncX, outIncY, outIncZ);

  // Loop through ouput pixels
  for (idxY = 0; idxY < dims[1]; idxY++) {
    for (idxR = 0; idxR < rowLength; idxR++) {
      // Pixel operation
      *outPtr = *inPtr;
      outPtr++;
      inPtr++;
    }
    outPtr += outIncY;
    inPtr += inIncY;
  }
}


/****************************************
*  Routines for Returning Data
*
*/
double vtkTrackedVideoData::GetTimeStamp(int num) {

  return this->Times->GetComponent(num,0);
}


vtkMatrix4x4 *vtkTrackedVideoData::GetDataElement(int num) {

  if (this->SensorMatrixCollection->GetNumberOfItems() > num) {
    return (vtkMatrix4x4 *)this->SensorMatrixCollection->GetItemAsObject(num);
  } else {
    return NULL;
  }
}


vtkDoubleArray *vtkTrackedVideoData::GetDataElementsAsArray(char *name) {

  int n,i,j;
  vtkDoubleArray *temp;

  if (strcmp(name, "times") == 0) {
    this->DataElementAsArray->DeepCopy(this->Times);
  } else {
    this->DataElementAsArray->SetNumberOfComponents(16);
  }
  if (strcmp(name, "registration") == 0) {
    this->DataElementAsArray->SetNumberOfTuples(1);
    for (i=0; i<4; i++) {
      for (j=0; j<4; j++) {
        this->DataElementAsArray->InsertComponent(0,4*i+j,this->RegMatrix->GetElement(i,j));
      }
    }
  }
  if (strcmp(name, "calibration") == 0) {
    temp->SetNumberOfTuples(1);
    for (i=0; i<4; i++) {
      for (j=0; j<4; j++) {
        this->DataElementAsArray->InsertComponent(0,4*i+j,this->CalibMatrix->GetElement(i,j));
      }
    }
  }
  if (strcmp(name, "sensordata") == 0) {
    this->DataElementAsArray->SetNumberOfTuples(this->GetNumElements());
    for (n=0; n<this->GetNumElements(); n++) {
      for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
          this->DataElementAsArray->InsertComponent(n,4*i+j,((vtkMatrix4x4 *)this->SensorMatrixCollection->GetItemAsObject(n))->GetElement(i,j));
        }
      }
    }
  }

  return this->DataElementAsArray;
}


vtkPoints *vtkTrackedVideoData::GetDataPositions()
{
  vtkMatrix4x4 *DataMatrix = vtkMatrix4x4::New();
  this->DataPositions->SetNumberOfPoints(this->SensorMatrixCollection->GetNumberOfItems());
  for (int num=0; num<this->SensorMatrixCollection->GetNumberOfItems(); num++) {
    DataMatrix->Multiply4x4((vtkMatrix4x4 *)this->SensorMatrixCollection->GetItemAsObject(num),this->CalibMatrix,DataMatrix);
    DataPositions->InsertPoint(num,DataMatrix->GetElement(0,3),DataMatrix->GetElement(1,3),DataMatrix->GetElement(2,3));
  }
  DataMatrix->Delete();
  return this->DataPositions;
}

vtkPoints *vtkTrackedVideoData::GetDataPositionsAfterRegistration()
{
  vtkMatrix4x4 *DataMatrix = vtkMatrix4x4::New();
  this->DataPositionsAfterRegistration->SetNumberOfPoints(this->SensorMatrixCollection->GetNumberOfItems());
  for (int num=0; num<this->SensorMatrixCollection->GetNumberOfItems(); num++) {
    DataMatrix->Multiply4x4((vtkMatrix4x4 *)this->SensorMatrixCollection->GetItemAsObject(num),this->CalibMatrix,DataMatrix);
    DataMatrix->Multiply4x4(this->RegMatrix,DataMatrix,DataMatrix);
    DataPositionsAfterRegistration->InsertPoint(num,DataMatrix->GetElement(0,3),DataMatrix->GetElement(1,3),DataMatrix->GetElement(2,3));
  }
  DataMatrix->Delete();
  return this->DataPositionsAfterRegistration;
}


int vtkTrackedVideoData::GetDataElementV(int num, vtkMatrix4x4 *buf, vtkImageData *imbuf) {

  if (this->SensorMatrixCollection->GetNumberOfItems() > num && this->ImageStored == 1) {
    buf = (vtkMatrix4x4 *)this->SensorMatrixCollection->GetItemAsObject(num);
    imbuf = (vtkImageData *)this->ImageCollection->GetItemAsObject(num);
    return 0;
  } else {
    return 1;
  }
}


vtkImageData *vtkTrackedVideoData::GetImage(int num) {

  if (this->ImageCollection->GetNumberOfItems() > num) {
    return (vtkImageData *)this->ImageCollection->GetItemAsObject(num);
  } else {
    return NULL;
  }
}


int vtkTrackedVideoData::GetNumElements() {

  return this->SensorMatrixCollection->GetNumberOfItems();
}


int vtkTrackedVideoData::GetNumImages() {

  if (this->NumPreallocFrames == 0) {
    return this->ImageCollection->GetNumberOfItems();
  } else {
    return this->NumPreallocFramesRecorded;
  }
}


int vtkTrackedVideoData::GetImageStored() {

  return this->ImageStored;
}


/****************************************
*  Routines for Writing Data Out
*
*/
int vtkTrackedVideoData::WriteToTxt(char *fname) {
  
  FILE *f;
  int n, i, j;
  char fn[500];
  vtkMatrix4x4 *temp;

  sprintf(fn,"%s.txt",fname);
  f = fopen(fn,"w");

  fprintf(f,"registration\n");
  temp = this->RegMatrix;
  for (i=0;i<4;i++) {
    for (j=0;j<4;j++) {
      fprintf(f,"%0.3f ",temp->GetElement(i,j));
    }
    fprintf(f,"\n");
  }
  fprintf(f,"\n");

  fprintf(f,"calibration\n");
  temp = this->CalibMatrix;
  for (i=0;i<4;i++) {
    for (j=0;j<4;j++) {
      fprintf(f,"%0.3f ",temp->GetElement(i,j));
    }
    fprintf(f,"\n");
  }
  fprintf(f,"\n");

  fprintf(f,"data\n");
  for (n=0;n<this->SensorMatrixCollection->GetNumberOfItems();n++) {
    temp = (vtkMatrix4x4 *)this->SensorMatrixCollection->GetItemAsObject(n);
    for (i=0;i<4;i++) {
      for (j=0;j<4;j++) {
        fprintf(f,"%0.3f ",temp->GetElement(i,j));
      }
      fprintf(f,"\n");
    }
    fprintf(f,"\n");
  }

  fprintf(f,"time_stamps\n");
  for (n=0;n<this->SensorMatrixCollection->GetNumberOfItems();n++) {
    fprintf(f,"%0.3f\n",this->GetTimeStamp(n));
  }
  fclose(f);


  //Save old position format
  sprintf(fn,"%s_mats.txt",fname);
  f = fopen(fn,"w");
  fprintf(f,"%d\n",this->SensorMatrixCollection->GetNumberOfItems());
  for (n=0;n<this->SensorMatrixCollection->GetNumberOfItems();n++) {
    temp = (vtkMatrix4x4 *)this->SensorMatrixCollection->GetItemAsObject(n);
    for (i=0;i<4;i++) {
      for (j=0;j<4;j++) {
        fprintf(f,"%0.3f ",temp->GetElement(i,j));
      }
      fprintf(f,"\n");
    }
  fprintf(f,"\n");
  }
  fclose(f);

  sprintf(fn,"%s_timestamps.txt",fname);
  f = fopen(fn,"w");
  fprintf(f,"%d\n",this->SensorMatrixCollection->GetNumberOfItems());
  for (n=0;n<this->SensorMatrixCollection->GetNumberOfItems();n++) {
    fprintf(f,"%0.3f\n",this->GetTimeStamp(n));
  }
  fclose(f);

  if (this->ImageStored) {
    vtkPNGWriter *writer = vtkPNGWriter::New();
    vtkImageMapToWindowLevelColors *map = vtkImageMapToWindowLevelColors::New();
    map->SetWindow(200);
    map->SetLevel(80);

    if (this->NumPreallocFrames == 0) {
      for (n=0;n<this->ImageCollection->GetNumberOfItems();n++) {
        sprintf(fn,"%s_img%03d.png",fname,n);
        writer->SetFileName(fn);
        //writer->SetInput((vtkImageData *)this->ImageCollection->GetItemAsObject(n));
        map->SetInput((vtkImageData *)this->ImageCollection->GetItemAsObject(n));
        writer->SetInput(map->GetOutput());
        writer->Write();
      }
    } else {
      // output preallocated frames
      vtkImageReslice *slice = vtkImageReslice::New();
      slice->SetResliceAxesDirectionCosines(1,0,0,0,1,0,0,0,1);
      slice->SetOutputSpacing(1.0,1.0,1.0);
      slice->SetOutputDimensionality(2);
      int *dims = this->prealloc_out->GetDimensions();
      slice->SetInput(this->prealloc_out);
      for (n=0; n<dims[2]; n++) {
        sprintf(fn,"%s_img%03d.png",fname,n);
        writer->SetFileName(fn);
        slice->SetResliceAxesOrigin(0,0,n);
        map->SetInput(slice->GetOutput());
        writer->SetInput(map->GetOutput());
        writer->Write();
      }
      slice->Delete();
    }

    writer->Delete();
  }

  return 1;
}


int vtkTrackedVideoData::WriteToVtk(char *fname) {

  vtkImageData *itemp, *itemp2 = NULL;
  vtkFieldData *field;

  vtkImageAppend *iappend = vtkImageAppend::New();
  vtkStructuredPointsWriter *writer = vtkStructuredPointsWriter::New();
  vtkDoubleArray *reg = vtkDoubleArray::New();
  vtkDoubleArray *calib = vtkDoubleArray::New();
  vtkDoubleArray *sensorData = vtkDoubleArray::New();

  int i,j,n;

  iappend->SetAppendAxis(2);
  reg->SetNumberOfComponents(16);
  calib->SetNumberOfComponents(16);
  sensorData->SetNumberOfComponents(16);

  if (this->NumPreallocFrames == 0) {
    for (n=0;n<this->ImageCollection->GetNumberOfItems();n++) {
      iappend->AddInput((vtkImageData *)this->ImageCollection->GetItemAsObject(n));
    }
    if (n==0) {
      itemp2 = vtkImageData::New();
      iappend->AddInput(itemp2);
    }
    iappend->Update();
    itemp = iappend->GetOutput();
  } else {
    itemp = this->prealloc_out;
  }

  for (i=0;i<4;i++) {
    for (j=0;j<4;j++) {
      reg->InsertComponent(0,j+4*i,this->RegMatrix->GetElement(i,j));
      calib->InsertComponent(0,j+4*i,this->CalibMatrix->GetElement(i,j));
    }
  }

  vtkMatrix4x4 *temp;
  for (n=0;n<this->SensorMatrixCollection->GetNumberOfItems();n++) {
    for (i=0;i<4;i++) {
      for (j=0;j<4;j++) {
        temp = (vtkMatrix4x4 *)this->SensorMatrixCollection->GetItemAsObject(n);
        sensorData->InsertComponent(n,j+4*i,temp->GetElement(i,j));
      }
    }
  }

  reg->SetName("Registration");
  calib->SetName("Calibration");
  sensorData->SetName("SensorPositions");
  Times->SetName("TimeStamps");

  field = itemp->GetFieldData();
  field->AddArray(reg);
  field->AddArray(calib);
  field->AddArray(sensorData);
  field->AddArray(Times);
  itemp->UpdateData();

  writer->SetFileName(fname);
  writer->SetInput(itemp);
  writer->SetFileTypeToBinary();
  writer->Write();

  iappend->Delete();
  writer->Delete();
  reg->Delete();
  calib->Delete();
  sensorData->Delete();

  if (itemp2 != NULL) {
    itemp2->Delete();
  }

  if (this->NumPreallocFrames > 0) {
    field->RemoveArray("Registration");
    field->RemoveArray("Calibration");
    field->RemoveArray("SensorPositions");
    field->RemoveArray("TimeStamps");
    itemp->UpdateData();
  }

  return 1;
}


/****************************************
*  Routines for Reading Data In
*
*/
int vtkTrackedVideoData::ReadFromTxt(char *fname) {
  
  // add code here
  return 1;
}


int vtkTrackedVideoData::ReadFromVtk(char *fname) {

  vtkImageData *itemp;
  vtkImageReslice *slice = vtkImageReslice::New();
  vtkFieldData *field;
  vtkStructuredPointsReader *reader = vtkStructuredPointsReader::New();
  vtkDoubleArray *reg = NULL;
  vtkDoubleArray *calib = NULL;
  vtkDoubleArray *sensorData = NULL;
  vtkDoubleArray *times = NULL;
  int dims[3], n, i, j;
  double sp[3];

  this->Reset();

  reader->SetFileName(fname);
  itemp = (vtkImageData *)reader->GetOutput();
  itemp->UpdateData();
  field = itemp->GetFieldData();
  reg = (vtkDoubleArray *)field->GetArray("Registration");
  calib = (vtkDoubleArray *)field->GetArray("Calibration");
  sensorData = (vtkDoubleArray *)field->GetArray("SensorPositions");
  times = (vtkDoubleArray *)field->GetArray("TimeStamps");

  // If sensor data or reg or calib are missing -> wrong file format
  if (sensorData == NULL || reg == NULL || calib == NULL)
  {
    slice->Delete();
    reader->Delete();
    return 0;
  }

  for (i=0;i<4;i++) {
    for (j=0;j<4;j++) {
      this->RegMatrix->SetElement(i,j,reg->GetComponent(0,j+4*i));
      this->CalibMatrix->SetElement(i,j,calib->GetComponent(0,j+4*i));
    }
  }

  itemp->GetDimensions(dims);
  itemp->GetSpacing(sp);
  slice->SetResliceAxesDirectionCosines(1,0,0,0,1,0,0,0,1);
  slice->SetOutputSpacing(sp[0],sp[1],1.0);
  slice->SetOutputDimensionality(2);

  vtkMatrix4x4 *temp = vtkMatrix4x4::New();
  for (n=0;n<sensorData->GetNumberOfTuples();n++) {
    for (i=0;i<4;i++) {
      for (j=0;j<4;j++) {
        temp->SetElement(i,j,sensorData->GetComponent(n,j+4*i));
      }
    }
    if (dims[2]==sensorData->GetNumberOfTuples()) {
      slice->SetInput(itemp);
//      slice->SetOutputExtent(0,dims[0],0,dims[1],0,0);
      slice->SetResliceAxesOrigin(0,0,n);
      this->AddDataElementV(times->GetComponent(n,0), temp, slice->GetOutput());
    } else {
      this->AddDataElement(times->GetComponent(n,0), temp);
    }
  }

  // Delete objects
  temp->Delete();
  slice->Delete();
  reader->Delete();
  return 1;
}

