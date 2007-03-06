/*=auto=========================================================================

  Portions (c) Copyright 2006 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFreeSurferProceduralColorNode.h,v $
  Date:      $Date: 2006/03/19 17:12:28 $
  Version:   $Revision: 1.0 $

=========================================================================auto=*/
// .NAME vtkMRMLFreeSurferProceduralColorNode - MRML node to represent FreeSurfer color information.
// .SECTION Description
// Color nodes describe colour look up tables. The tables are pre-generated by
// this node to provide the default FS look up tables (heat, red/green,
// blue/red etc). More than one model or label volume or editor can access the prebuilt
// nodes.

#ifndef __vtkMRMLFreeSurferProceduralColorNode_h
#define __vtkMRMLFreeSurferProceduralColorNode_h

#include <string>
#include <vector>

#include "vtkMRML.h"
#include "vtkMRMLNode.h"
#include "vtkFSLookupTable.h"
#include "vtkMRMLProceduralColorNode.h"

class vtkLookupTable;
class vtkFSLookupTable;
class VTK_MRML_EXPORT vtkMRMLFreeSurferProceduralColorNode : public vtkMRMLProceduralColorNode
{
public:
  static vtkMRMLFreeSurferProceduralColorNode *New();
  vtkTypeMacro(vtkMRMLFreeSurferProceduralColorNode,vtkMRMLColorNode);
  void PrintSelf(ostream& os, vtkIndent indent);
  
  //--------------------------------------------------------------------------
  // MRMLNode methods
  //--------------------------------------------------------------------------

  virtual vtkMRMLNode* CreateNodeInstance();

  // Description:
  // Set node attributes
  virtual void ReadXMLAttributes( const char** atts);

  // Description:
  // Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);


  // Description:
  // Read in a text file from FreeSurfer holding colour table elements 
  // id<white space>name<white-space>r<w-s>g<w-s>b<w-s>a\n
  // comments start with a hash mark
  virtual void ReadFile ();
  
  // Description:
  // Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node);
  
  // Description:
  // Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() {return "FreeSurferProceduralColor";};

  // Description:
  // 
  virtual void UpdateScene(vtkMRMLScene *scene);

  // Description:
  // Set the lookup table as a freesurfer lookup table, and get it as a
  // vtkLookupTable or an vtkFSLookupTable
  vtkLookupTable *GetLookupTable();
  vtkFSLookupTable *GetFSLookupTable();
  vtkSetObjectMacro(LookupTable, vtkFSLookupTable);

  // Description:
  // Get/Set for Type
  void SetType(int type);
  void SetTypeToHeat();
  void SetTypeToBlueRed();
  void SetTypeToRedBlue();
  void SetTypeToRedGreen();
  void SetTypeToGreenRed();
  void SetTypeToLabels();
  void SetTypeToFile();

  void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData );

  //BTX
  // Description:
  // The list of valid table types
 
  // Heat - 
  // BlueRed - 
  // RedBlue - reversed BlueRed
  // RedGreen - 
  // GreenRed - reversed RedGreen
  enum
  {
    Heat = 1,
    BlueRed = 2,
    RedBlue = 3,
    RedGreen = 4,
    GreenRed = 5,
    Labels = 6,
    File = 7,
  };
  //ETX

  // Description:
  // Return the lowest and the highest integers, for use in looping
  int GetFirstType() { return this->Heat; };
  int GetLastType () { return this->File; };
  
  // Description:
  // return a text string describing the colour look up table type
  const char * GetTypeAsString();

  // Description:
  // return a text string describing the class name and type name
  const char * GetTypeAsIDString();
  
  //BTX
  // Description:
  // DisplayModifiedEvent is generated when display node parameters is changed
  enum
    {
      DisplayModifiedEvent = 20000,
    };
//ETX
  
protected:
  vtkMRMLFreeSurferProceduralColorNode();
  ~vtkMRMLFreeSurferProceduralColorNode();
  vtkMRMLFreeSurferProceduralColorNode(const vtkMRMLFreeSurferProceduralColorNode&);
  void operator=(const vtkMRMLFreeSurferProceduralColorNode&);

  // Description:
  // Set values in the names vector from the colour rgba entries in the colour
  // table
  void SetNamesFromColors();

  // Description:
  // a lookup table tailored with FreeSurfer colours, constructed according to Type
  vtkFSLookupTable *LookupTable;
};

#endif
