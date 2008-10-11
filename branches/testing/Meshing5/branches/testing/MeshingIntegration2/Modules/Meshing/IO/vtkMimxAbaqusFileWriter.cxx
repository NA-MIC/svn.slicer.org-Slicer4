/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkMimxAbaqusFileWriter.cxx,v $
Language:  C++
Date:      $Date: 2008/04/27 03:34:28 $
Version:   $Revision: 1.18 $

 Musculoskeletal Imaging, Modelling and Experimentation (MIMX)
 Center for Computer Aided Design
 The University of Iowa
 Iowa City, IA 52242
 http://www.ccad.uiowa.edu/mimx/
 
Copyright (c) The University of Iowa. All rights reserved.
See MIMXCopyright.txt or http://www.ccad.uiowa.edu/mimx/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "vtkMimxAbaqusFileWriter.h"

#include <vtksys/SystemTools.hxx>
#include "vtkPolyData.h"
#include "vtkUnstructuredGrid.h"
#include "vtkUnstructuredGridReader.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkIntArray.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPointSet.h"
#include "vtkPoints.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkStringArray.h"
#include <string.h>
#include <list>
using std::list;


vtkCxxRevisionMacro(vtkMimxAbaqusFileWriter, "$Revision: 1.18 $");
vtkStandardNewMacro(vtkMimxAbaqusFileWriter);

vtkMimxAbaqusFileWriter::vtkMimxAbaqusFileWriter()
{
  HeaderInformation = "";
  FileName = "";
  NodeElementFileName = "";
}

vtkMimxAbaqusFileWriter::~vtkMimxAbaqusFileWriter()
{
}


//----------------------------------------------------------------------------
int vtkMimxAbaqusFileWriter::RequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  // get the info objects
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  // get the input and output
  vtkDataSet *input = vtkDataSet::SafeDownCast(
    inInfo->Get(vtkDataObject::DATA_OBJECT()));
    
  if (! vtkUnstructuredGrid::SafeDownCast(input)->GetPointData()->GetArray("Node_Numbers") )
  {
    vtkErrorMacro("Unstructured grid must contain field data called 'Node_Numbers'");
    return 0;
  }
  
  if (! vtkUnstructuredGrid::SafeDownCast(input)->GetCellData()->GetArray("Element_Numbers") )
  {
    vtkErrorMacro("Unstructured grid must contain field data called 'Element_Numbers'");
    return 0;
  }

  std::ofstream outfile;  
  outfile.open(this->FileName.c_str(), std::ios::out );

  // GetFilenameWithoutExtension
  if (NodeElementFileName.length() == 0)
  {
    this->NodeElementFileName = vtksys::SystemTools::GetFilenamePath( this->FileName );
        std::string fname = vtksys::SystemTools::GetFilenameWithoutExtension(this->FileName);
        if(this->NodeElementFileName.length() != 0)
        {
                this->NodeElementFileName += "/";
        }
        this->NodeElementFileName += fname;
    this->NodeElementFileName += "_input";
  }
  std::ofstream nodefile;  
  nodefile.open(this->NodeElementFileName.c_str(), std::ios::out );


  WriteHeader ( outfile );
  WriteHeadingSection( outfile );
  WriteNodeElementHeader( outfile );
  WriteNodes( nodefile, vtkUnstructuredGrid::SafeDownCast( input ) );
  WriteHexElements( nodefile, vtkUnstructuredGrid::SafeDownCast( input ) );
  WriteTetElements( nodefile, vtkUnstructuredGrid::SafeDownCast( input ) );
  WriteQuadElements( nodefile, vtkUnstructuredGrid::SafeDownCast( input ) );
  WriteNodeSets(nodefile, vtkUnstructuredGrid::SafeDownCast( input ) );
  //Get Unique Material Properties
  WriteElementSets( nodefile, vtkUnstructuredGrid::SafeDownCast( input ) );
  WriteMaterialHeader( outfile );
  WriteMaterialProperties( outfile, vtkUnstructuredGrid::SafeDownCast( input ));
  WriteBoundaryConditions( outfile, vtkUnstructuredGrid::SafeDownCast( input ) );
  WriteFooter( outfile );

  return 1;
}

  
void vtkMimxAbaqusFileWriter::WriteHeader( ostream& os )
{
  os << "**=========================================================================" << std::endl;
  os << "**     MODEL:" << std::endl;
  os << "**        ==>" << this->FileName << std::endl; 
  os << "**" << std::endl;
  if ( this->HeaderInformation.length() > 0 )
  {
    os << "**     USER INFORMATION:" << std::endl;
    int strIndex = HeaderInformation.find("\n", 0);
    if (strIndex != std::string::npos)
    {
      os << "**                " << 
        this->HeaderInformation.substr(0,strIndex) << 
        std::endl;
      while (strIndex != std::string::npos)
      {
        int prevIndex = strIndex + 1;
        strIndex = HeaderInformation.find("\n", prevIndex);
        if (strIndex == std::string::npos)
        {
          if (strIndex != this->HeaderInformation.length())
            os << "**                " << 
                 this->HeaderInformation.substr(prevIndex,this->HeaderInformation.length()-prevIndex+1) << 
                 std::endl;
        }
        else if (strIndex == prevIndex)
        {
          os << "**                " << std::endl;
        }
        else
        {
          os << "**                " << 
            this->HeaderInformation.substr(prevIndex,strIndex-prevIndex) << 
            std::endl;
        }
      }
    }
    else
    {
      os << "**                " << this->HeaderInformation << std::endl;
    }
  }
  
  os << "**" << std::endl;
  os << "**" << std::endl;
  os << "**     CREATED ON:" << std::endl;
  // Get Date - vtksys::SystemTools::GetCurrentDateTime
  std::string dateString = vtksys::SystemTools::GetCurrentDateTime("%m/%d/%Y %H:%M");
  // char * asctime(const struct tm* tm);
  // time()
  os << "**        ==> " << dateString << std::endl;
  os << "**" << std::endl;
  /*
  os << "**     RUN COMMAND:" << std::endl;
  os << "**        ==> abaqus job=filename datacheck" << std::endl; 
  os << "**        ==> abaqus job=filename" << std::endl;
  os << "**" << std::endl;
  os << "**" << std::endl;
  os << "**     DESCRIPTION:  Include a brief description of the model" << std::endl;
  os << "**" << std::endl;
  */
  os << "**     MODIFICATION HISTORY:" << std::endl;
  os << "**        ==>" << std::endl;
  os << "**" << std::endl;
  os << "**=========================================================================" << std::endl;
  os << "**                            MODEL DATA                                 **" << std::endl;
}

void vtkMimxAbaqusFileWriter::WriteHeadingSection( ostream& os )
{
/*
  if ( Heading.length() )
  {
    os << "*HEADING" << endl;
    os << Heading << endl;
  }
*/
}

void vtkMimxAbaqusFileWriter::WriteNodeElementHeader( ostream& os )
{
  os << "**=========================================================================" << std::endl;
  os << "**                                               Node / Element Definitions" << std::endl;
  os << "**=========================================================================" << std::endl;
  os << "*INCLUDE, INPUT=" << this->NodeElementFileName << std::endl;
}    


void vtkMimxAbaqusFileWriter::WriteNodes( ostream& os, vtkUnstructuredGrid *grid )
{
  vtkIntArray *nodeArray = vtkIntArray::SafeDownCast( grid->GetPointData()->GetArray("Node_Numbers") );
  int numberOfPoints = grid->GetNumberOfPoints();
  
  os << "*NODE" << std::endl;
  
  for( int pointIndex = 0; pointIndex < numberOfPoints; pointIndex++ )
  {
    double point[3];
    grid->GetPoint( pointIndex, point );
    os << nodeArray->GetValue(pointIndex) << ", " << point[0] << ", " << point[1] << ", " << point[2] << std::endl;
  }
}

void vtkMimxAbaqusFileWriter::WriteHexElements( ostream& os, vtkUnstructuredGrid *grid )
{  
  vtkIntArray *nodeArray = vtkIntArray::SafeDownCast( grid->GetPointData()->GetArray("Node_Numbers") );
  vtkIntArray *elementArray = vtkIntArray::SafeDownCast( grid->GetCellData()->GetArray("Element_Numbers") );
  
  int numberOfCells = grid->GetNumberOfCells( );
  bool headerFlag = true;
  
  for( int elementId = 0; elementId < numberOfCells; elementId++ )
  {
    int cellType = grid->GetCellType( elementId );
    if ( cellType == VTK_HEXAHEDRON )
    {
      if ( headerFlag )
      {
        os << "**" << std::endl;
        os << "**" << std::endl;
        os << "*ELEMENT, TYPE=C3D8" << std::endl;
        headerFlag = false;
      }
      
      vtkIdList *idlist = grid->GetCell( elementId )->GetPointIds( );
      int numberOfIds = idlist->GetNumberOfIds( );
      
      os << elementArray->GetValue(elementId);
      
      for ( int count = 0; count < numberOfIds; count ++ )
      {
        os << ", " << nodeArray->GetValue( idlist->GetId(count) );
      }
      os << std::endl;
    }
  }
}

                          
void vtkMimxAbaqusFileWriter::WriteTetElements( ostream& os, vtkUnstructuredGrid *grid )
{  
  vtkIntArray *nodeArray = vtkIntArray::SafeDownCast( grid->GetPointData()->GetArray("Node_Numbers") );
  vtkIntArray *elementArray = vtkIntArray::SafeDownCast( grid->GetCellData()->GetArray("Element_Numbers") );

  int numberOfCells = grid->GetNumberOfCells( );  
  bool headerFlag = false;
  
  for( int elementId = 0; elementId < numberOfCells; elementId++ )
  {
    int cellType = grid->GetCellType( elementId );
    if ( cellType == VTK_TETRA )
    {
      if ( headerFlag )
      {
        os << "**" << std::endl;
        os << "**" << std::endl;
        os << "*ELEMENT, TYPE=C3D4" << std::endl;
        headerFlag = false;
      }
      vtkIdList *idlist = grid->GetCell( elementId )->GetPointIds( );
      int numberOfIds = idlist->GetNumberOfIds( );
      
      os << elementArray->GetValue(elementId);
      
      for ( int count = 0; count < numberOfIds; count ++ )
      {
        os << ", " << nodeArray->GetValue( idlist->GetId(count) );
      }
      os << std::endl;
    }
  }
}

void vtkMimxAbaqusFileWriter::WriteQuadElements( ostream& os, vtkUnstructuredGrid *grid )
{  
  vtkIntArray *nodeArray = vtkIntArray::SafeDownCast( grid->GetPointData()->GetArray("Node_Numbers") );
  vtkIntArray *elementArray = vtkIntArray::SafeDownCast( grid->GetCellData()->GetArray("Element_Numbers") );
  
  int numberOfCells = grid->GetNumberOfCells( );
  int headerFlag = true;
  
  for( int elementId = 0; elementId < numberOfCells; elementId++ )
  {
    int cellType = grid->GetCellType( elementId );
    if ( cellType == VTK_QUAD )
    {
      if ( headerFlag )
      {
        os << "**" << std::endl;
        os << "**" << std::endl;
        os << "*ELEMENT, TYPE=R3D4" << std::endl;
        headerFlag = false;
      }
      vtkIdList *idlist = grid->GetCell( elementId )->GetPointIds( );
      int numberOfIds = idlist->GetNumberOfIds( );
      
      os << elementArray->GetValue(elementId);
      
      for ( int count = 0; count < numberOfIds; count ++ )
      {
        os << ", " << nodeArray->GetValue( idlist->GetId(count) );
      }
      os << std::endl;
    }
  }
}

void vtkMimxAbaqusFileWriter::WriteElementSets( ostream& os, vtkUnstructuredGrid *grid )
{  
        // loop through all element sets
        int i,j;
        int numelementarrays = grid->GetCellData()->GetNumberOfArrays();
        vtkIntArray *elementnumarray = vtkIntArray::SafeDownCast(
                grid->GetCellData()->GetArray("Element_Numbers"));
        for (i=0; i<numelementarrays; i++)
        {
                vtkIntArray *intarray = vtkIntArray::SafeDownCast(grid->GetCellData()->GetArray(i));
                if(intarray && strcmp(grid->GetCellData()->GetArray(i)->GetName(), "Element_Numbers")
                        && intarray->GetNumberOfComponents() == 1)
                {
                        list<int> elementnum;
                        elementnum.clear();
                        // add the nodes belonging to the node set to the list
                        for (j=0; j<intarray->GetNumberOfTuples(); j++)
                        {
                                if(intarray->GetValue(j))
                                {
                                        elementnum.push_back(elementnumarray->GetValue(j));
                                }
                        }
                        if (!elementnum.size())
                        {
                                return;
                        }
                        // sort the entries in ascending order
                        elementnum.sort();                      
                        vtkIdList *storearray = vtkIdList::New();
                        do 
                        {
                                storearray->InsertUniqueId(elementnum.front());
                                elementnum.pop_front();
                        } while(elementnum.size());

                        int numberOfElements = storearray->GetNumberOfIds();
                        const char *arrayname = grid->GetCellData()->GetArray(i)->GetName();
                        os << "*ELSET, ELSET=" << arrayname << ", GENERATE" << std::endl;
                        // loop through all the nodes in the node set
                        int firstelement, lastelement, currelement;
                        if(numberOfElements > 1)
                        {
                                for (j=0; j<numberOfElements; j++)
                                {
                                        if(j == 0)
                                        {
                                                firstelement = storearray->GetId(j);
                                                lastelement = firstelement;
                                                currelement = storearray->GetId(j+1);
                                                j++;
                                        }
                                        if(currelement != lastelement+1)
                                        {
                                                os<<firstelement<<", "<<lastelement<<std::endl;
                                                firstelement = currelement;
                                                lastelement = currelement;
                                                if(j < numberOfElements-1)
                                                {
                                                        currelement = storearray->GetId(j+1);
                                                }
                                                if (j == numberOfElements-1)
                                                {
                                                        os<<firstelement<<", "<<lastelement<<std::endl;
                                                }
                                        }
                                        else
                                        {
                                                lastelement = currelement;
                                                if(j < numberOfElements-1)
                                                {
                                                        currelement = storearray->GetId(j+1);
                                                }
                                                if(j == numberOfElements-1)
                                                {
                                                        os<<firstelement<<", "<<lastelement<<std::endl;
                                                }
                                        }
                                }
                        }
                        else{
                                firstelement = storearray->GetId(0);
                                lastelement = storearray->GetId(0);
                                os<<firstelement<<", "<<lastelement<<std::endl;
                        }
                }
        } 
}

void vtkMimxAbaqusFileWriter::WriteMaterialHeader( ostream& os )
{
  os << "**" << std::endl;
  os << "**=========================================================================" << std::endl;
  os << "**                                                    Material  Definitions" << std::endl;
  os << "**=========================================================================" << std::endl;
}
    
void vtkMimxAbaqusFileWriter::WriteMaterialProperties( ostream& os, vtkUnstructuredGrid *grid )
{
        int i;

        vtkStringArray *strarray = vtkStringArray::SafeDownCast(
                grid->GetFieldData()->GetAbstractArray("Element_Set_Names"));

        if(!strarray)
        {
                return;
        }
        int numEntries = strarray->GetNumberOfTuples();

        for (i=0; i<numEntries; i++)
        {
                char str[256];
                strcpy(str, strarray->GetValue(i));
                strcat(str, "_Constant_Youngs_Modulus");
                vtkFloatArray *youngarray = vtkFloatArray::SafeDownCast(
                        grid->GetFieldData()->GetArray(str));

                char strpoi[256];
                strcpy(strpoi, strarray->GetValue(i));
                strcat(strpoi, "_Constant_Poissons_Ratio");
                vtkFloatArray *poissonarray = vtkFloatArray::SafeDownCast(
                        grid->GetFieldData()->GetArray(strpoi));

                if(youngarray && poissonarray)
                {
                        float youngsmodulus = youngarray->GetValue(0);
                        float poissonsratio = poissonarray->GetValue(0);
                        os << "*SOLID SECTION, ELSET=" << strarray->GetValue(i) 
                                << ", MATERIAL=" << strarray->GetValue(i) << std::endl;
                        os << "*MATERIAL, NAME=" << strarray->GetValue(i) << std::endl; 
                        os << "*ELASTIC" << std::endl;
                        os << youngsmodulus << ", " << poissonsratio << std::endl;      
                }
                else
                {
                        if(poissonarray)
                        {
                                vtkIntArray *elementArray = vtkIntArray::SafeDownCast( 
                                        grid->GetCellData()->GetArray(strarray->GetValue(i)) );

                                int numberOfCells = grid->GetNumberOfCells( );
                                
                                char str[256];
                                strcpy(str, strarray->GetValue(i));
                                strcat(str, "_Image_Based_Material_Property");
                                
                                char rebinstr[256];
                                strcpy(rebinstr, str);
                                strcat(rebinstr, "_ReBin");
                                // store the elements with repeating material properties
                                vtkDoubleArray *materialArray  = vtkDoubleArray::SafeDownCast( 
                                        grid->GetCellData()->GetArray(rebinstr));
                                if(!materialArray)      materialArray  = vtkDoubleArray::SafeDownCast( 
                                        grid->GetCellData()->GetArray(str));
                                if(materialArray)
                                        this->WriteRepeatingMaterialProperties(
                                                elementArray, materialArray, grid, os, 
                                                strarray->GetValue(i), poissonarray->GetValue(0));
                        }
                }
        }
}

void vtkMimxAbaqusFileWriter::WriteBoundaryConditions( ostream& os, vtkUnstructuredGrid *grid )
{  
  
  os << "**=========================================================================" << std::endl;
  os << "**                               HISTORY DATA                            **" << std::endl;
  os << "**=========================================================================" << std::endl;
  char Concatenate[256];
  int i, j;
  vtkStringArray *nodesetnamestring = vtkStringArray::SafeDownCast(
          grid->GetFieldData()->GetAbstractArray("Node_Set_Names"));
  if(!nodesetnamestring)        return;

  vtkIntArray *NumSteps = vtkIntArray::SafeDownCast(
          grid->GetFieldData()->GetArray("Boundary_Condition_Number_Of_Steps"));
  if(!NumSteps) return;
  for(j=0; j<NumSteps->GetValue(0); j++)
  {
          char stepnum[256];
          sprintf(stepnum, "%d", j+1);
          //itoa(j+1, stepnum, 10);
          os << "*STEP" << std::endl;
          os << "*STATIC" << std::endl;
          os << "*BOUNDARY, OP=NEW" << std::endl;
          os << "*CLOAD" << std::endl;
          vtkFloatArray *floatarray;
          for(i=0; i<nodesetnamestring->GetNumberOfValues(); i++)
          {
                  const char* nodesetname =  nodesetnamestring->GetValue(i);

                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Force", "X", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 1, "<<floatarray->GetValue(0) << std::endl;
                  }
                  //
                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Force", "Y", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 2, "<<floatarray->GetValue(0) << std::endl;
                  }
                  //
                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Force", "Z", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 3, "<<floatarray->GetValue(0) << std::endl;
                  }
                  //
                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Displacement", "X", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 1, 1, "<<floatarray->GetValue(0) << std::endl;
                  }
                  //
                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Displacement", "Y", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 2, 2, "<<floatarray->GetValue(0) << std::endl;
                  }
                  //
                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Displacement", "Z", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 3, 3, "<<floatarray->GetValue(0) << std::endl;
                  }
                  //
                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Rotation", "X", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 4, 4, "<<floatarray->GetValue(0) << std::endl;
                  }
                  //
                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Rotation", "Y", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 5, 5, "<<floatarray->GetValue(0) << std::endl;
                  }
                  //
                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Rotation", "Z", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 6, 6, "<<floatarray->GetValue(0) << std::endl;
                  }
                  //
                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Moment", "X", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 4, "<<floatarray->GetValue(0) << std::endl;
                  }
                  //
                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Moment", "Y", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 5, "<<floatarray->GetValue(0) << std::endl;
                  }
                  //
                  Concatenate[256];
                  this->ConcatenateStrings("Step", stepnum, nodesetname, "Moment", "Z", Concatenate);
                  floatarray = vtkFloatArray::SafeDownCast(
                          grid->GetFieldData()->GetArray(Concatenate));
                  if(floatarray)
                  {
                          os << nodesetname<<", 6, "<<floatarray->GetValue(0) << std::endl;
                  }
          }
  }
}
//---------------------------------------------------------------------------------------------------
void vtkMimxAbaqusFileWriter::WriteNodeSets(ostream& os, vtkUnstructuredGrid *grid )
{
        // loop through all node sets
        int i,j;
        int numnodearrays = grid->GetPointData()->GetNumberOfArrays();
        vtkIntArray *nodenumarray = vtkIntArray::SafeDownCast(
                grid->GetPointData()->GetArray("Node_Numbers"));
        for (i=0; i<numnodearrays; i++)
        {
                vtkIntArray *intarray = vtkIntArray::SafeDownCast(grid->GetPointData()->GetArray(i));
                if(intarray && strcmp(grid->GetPointData()->GetArray(i)->GetName(), "Node_Numbers")
                        && intarray->GetNumberOfComponents() == 1)
                {
                        list<int> nodenum;
                        nodenum.clear();
                        // add the nodes belonging to the node set to the list
                        for (j=0; j<intarray->GetNumberOfTuples(); j++)
                        {
                                if(intarray->GetValue(j))
                                {
                                        nodenum.push_back(nodenumarray->GetValue(j));
                                }
                        }
                        if(!nodenum.size())
                        {
                                return;
                        }
                        // sort the entries in ascending order
                        nodenum.sort();                 
                        vtkIdList *storearray = vtkIdList::New();
                        do 
                        {
                                storearray->InsertUniqueId(nodenum.front());
                                nodenum.pop_front();
                        } while(nodenum.size());
        
                        int numberOfNodes = storearray->GetNumberOfIds();
                        const char *arrayname = grid->GetPointData()->GetArray(i)->GetName();
                        os << "*NSET, NSET=" << arrayname << ", GENERATE" << std::endl;
                        // loop through all the nodes in the node set
                        int firstnode, lastnode, currnode;
                        if(numberOfNodes > 1)
                        {
                                for (j=0; j<numberOfNodes; j++)
                                {
                                        if(j == 0)
                                        {
                                                firstnode = storearray->GetId(j);
                                                lastnode = firstnode;
                                                currnode = storearray->GetId(j+1);
                                                j++;
                                        }
                                        if(currnode != lastnode+1)
                                        {
                                                os<<firstnode<<", "<<lastnode<<std::endl;
                                                firstnode = currnode;
                                                lastnode = currnode;
                                                if(j < numberOfNodes-1)
                                                {
                                                        currnode = storearray->GetId(j+1);
                                                }
                                                if (j == numberOfNodes-1)
                                                {
                                                        os<<firstnode<<", "<<lastnode<<std::endl;
                                                }
                                        }
                                        else
                                        {
                                                lastnode = currnode;
                                                if(j < numberOfNodes-1)
                                                {
                                                        currnode = storearray->GetId(j+1);
                                                }
                                                if(j == numberOfNodes-1)
                                                {
                                                        os<<firstnode<<", "<<lastnode<<std::endl;
                                                }
                                        }
                                }
                        }
                        else{
                                firstnode = storearray->GetId(0);
                                lastnode = storearray->GetId(0);
                                os<<firstnode<<", "<<lastnode<<std::endl;
                        }
                }
        }
}
//---------------------------------------------------------------------------------------------------
void vtkMimxAbaqusFileWriter::WriteRepeatingMaterialProperties(
        vtkIntArray *ElementIds, vtkDoubleArray *MatProp, 
        vtkUnstructuredGrid *grid , ostream& os, 
        const char *ElementSetName, double PoissonsRatio)
{
        vtkIntArray *elementArray = vtkIntArray::SafeDownCast( 
                grid->GetCellData()->GetArray("Element_Numbers") );
        int i, j;
        int numEle = elementArray->GetNumberOfTuples();

        //list all the unique material properties
        vtkDoubleArray *UniqueMatProp = vtkDoubleArray::New();

        for (i=0; i<MatProp->GetNumberOfTuples(); i++)
        {
                if(MatProp->GetValue(i) >= 0.0)
                {
                        bool status = true;
                        for (j=0; j<UniqueMatProp->GetNumberOfTuples(); j++)
                        {
                                if(UniqueMatProp->GetValue(j) == MatProp->GetValue(i))
                                {
                                        status = false;
                                        break;
                                }
                        }
                        if(status)      
                        {
                                UniqueMatProp->InsertNextValue(MatProp->GetValue(i));
                        }
                }
        }
        // based on unique material properties generate element sets.
        for (i=0; i<UniqueMatProp->GetNumberOfTuples(); i++)
        {
                vtkIntArray *intarray = vtkIntArray::New();
                for (j=0; j<MatProp->GetNumberOfTuples(); j++)
                {
                        if (UniqueMatProp->GetValue(i) == MatProp->GetValue(j))
                        {
                                intarray->InsertNextValue(elementArray->GetValue(j));
                        }
                }
                if(intarray->GetNumberOfTuples() > 0)
                {
                        list<int> elementnum;
                        elementnum.clear();
                        // add the nodes belonging to the node set to the list
                        for (j=0; j<intarray->GetNumberOfTuples(); j++)
                        {
                                elementnum.push_back(intarray->GetValue(j));
                        }
                        if (!elementnum.size())
                        {
                                return;
                        }
                        // sort the entries in ascending order
                        elementnum.sort();                      
                        vtkIdList *storearray = vtkIdList::New();
                        do 
                        {
                                storearray->InsertUniqueId(elementnum.front());
                                elementnum.pop_front();
                        } while(elementnum.size());

                        int numberOfElements = storearray->GetNumberOfIds();
                        os << "*ELSET, ELSET=" << ElementSetName<<"_"<<i << ", GENERATE" << std::endl;
                        // loop through all the nodes in the node set
                        int firstelement, lastelement, currelement;
                        if(numberOfElements > 1)
                        {
                                for (j=0; j<numberOfElements; j++)
                                {
                                        if(j == 0)
                                        {
                                                firstelement = storearray->GetId(j);
                                                lastelement = firstelement;
                                                currelement = storearray->GetId(j+1);
                                                j++;
                                        }
                                        if(currelement != lastelement+1)
                                        {
                                                os<<firstelement<<", "<<lastelement<<std::endl;
                                                firstelement = currelement;
                                                lastelement = currelement;
                                                if(j < numberOfElements-1)
                                                {
                                                        currelement = storearray->GetId(j+1);
                                                }
                                                if (j == numberOfElements-1)
                                                {
                                                        os<<firstelement<<", "<<lastelement<<std::endl;
                                                }
                                        }
                                        else
                                        {
                                                lastelement = currelement;
                                                if(j < numberOfElements-1)
                                                {
                                                        currelement = storearray->GetId(j+1);
                                                }
                                                if(j == numberOfElements-1)
                                                {
                                                        os<<firstelement<<", "<<lastelement<<std::endl;
                                                }
                                        }
                                }
                        }
                        else{
                                firstelement = storearray->GetId(0);
                                lastelement = storearray->GetId(0);
                                os<<firstelement<<", "<<lastelement<<std::endl;
                        }
                        storearray->Delete();
                        os << "*SOLID SECTION, ELSET=" << ElementSetName<<"_"<<i 
                                << ", MATERIAL=" << ElementSetName<<"_"<<i << std::endl;
                        os << "*MATERIAL, NAME=" << ElementSetName<<"_"<<i << std::endl; 
                        os << "*ELASTIC" << std::endl;
                        os << UniqueMatProp->GetValue(i) << ", " << PoissonsRatio << std::endl; 

                }
                intarray->Delete();
        }
        //
        UniqueMatProp->Delete();
}
//---------------------------------------------------------------------------------------------------
void vtkMimxAbaqusFileWriter::WriteFooter( ostream& os )
{
  os << "**" << std::endl;
  os << "**" << std::endl;
  os << "*ENDSTEP" << std::endl;
  os << "**" << std::endl;
  os << "**==========================================================================" << std::endl;
  os << "**                         END OF ABAQUS INPUT DECK                       **" << std::endl;
  os << "**==========================================================================" << std::endl;
}
//---------------------------------------------------------------------------------------------------
void vtkMimxAbaqusFileWriter::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//---------------------------------------------------------------------------------------------------
void vtkMimxAbaqusFileWriter::ConcatenateStrings(const char* Step, const char* Num, 
                const char* NodeSetName, const char* Type, const char* Direction, char *Name)
{
        strcpy(Name, Step);
        strcat(Name, "_");
        strcat(Name,Num);
        strcat(Name, "_");
        strcat(Name, NodeSetName);
        strcat(Name, "_");
        strcat(Name,Type);
        strcat(Name, "_");
        strcat(Name, Direction);
}
//-----------------------------------------------------------------------------------------------------



