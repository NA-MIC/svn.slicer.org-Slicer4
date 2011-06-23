/*=auto=========================================================================

  Portions (c) Copyright 2006 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFreeSurferProceduralColorNode.h,v $
  Date:      $Date: 2006/03/19 17:12:28 $
  Version:   $Revision: 1.0 $

=========================================================================auto=*/
///  vtkMRMLFreeSurferProceduralColorNode - MRML node to represent FreeSurfer color information.
/// 
/// Color nodes describe colour look up tables. The tables are pre-generated by
/// this node to provide the default FS look up tables (heat, red/green,
/// blue/red etc). More than one model or label volume or editor can access the prebuilt
/// nodes.

#ifndef __vtkMRMLFreeSurferProceduralColorNode_h
#define __vtkMRMLFreeSurferProceduralColorNode_h

#include "vtkMRML.h"
#include "vtkMRMLNode.h"
#include "vtkMRMLProceduralColorNode.h"

class vtkLookupTable;
class vtkFSLookupTable;
class VTK_MRML_EXPORT vtkMRMLFreeSurferProceduralColorNode : public vtkMRMLProceduralColorNode
{
public:
  static vtkMRMLFreeSurferProceduralColorNode *New();
  vtkTypeMacro(vtkMRMLFreeSurferProceduralColorNode,vtkMRMLProceduralColorNode);
  void PrintSelf(ostream& os, vtkIndent indent);
  
  //--------------------------------------------------------------------------
  /// MRMLNode methods
  //--------------------------------------------------------------------------

  virtual vtkMRMLNode* CreateNodeInstance();

  /// 
  /// Set node attributes
  virtual void ReadXMLAttributes( const char** atts);

  /// 
  /// Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);
  
  /// 
  /// Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node);
  
  /// 
  /// Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() {return "FreeSurferProceduralColor";};

  /// 
  /// 
  virtual void UpdateScene(vtkMRMLScene *scene);

  /// 
  /// Set the lookup table as a freesurfer lookup table, and get it as a
  /// vtkLookupTable or an vtkFSLookupTable
  vtkLookupTable *GetLookupTable();
  vtkFSLookupTable *GetFSLookupTable();
  virtual void SetLookupTable(vtkFSLookupTable* newLookupTable);
  virtual vtkScalarsToColors* GetScalarsToColors();

  /// 
  /// Get/Set for Type
  void SetType(int type);
  void SetTypeToHeat();
  void SetTypeToBlueRed();
  void SetTypeToRedBlue();
  void SetTypeToRedGreen();
  void SetTypeToGreenRed();
  void SetTypeToLabels();
  void SetTypeToCustom();
  
  void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData );

  /// 
  /// The list of valid table types
 
  /// Heat - 
  /// BlueRed - 
  /// RedBlue - reversed BlueRed
  /// RedGreen - 
  /// GreenRed - reversed RedGreen
  /// Labels - info not held in this node, used for creating ids when read
  /// labels file into a vtkMRMLColorTableNode
  enum
  {
    Heat = 1,
    BlueRed = 2,
    RedBlue = 3,
    RedGreen = 4,
    GreenRed = 5,
    Labels = 6,
    Custom = 7
  };

  /// 
  /// Return the lowest and the highest integers, for use in looping (don't
  /// create labels)
  int GetFirstType() { return this->Heat; };
  int GetLastType () { return this->GreenRed; };
  
  /// 
  /// return a text string describing the colour look up table type
  const char * GetTypeAsString();

  /// 
  /// return a text string describing the class name and type name
  const char * GetTypeAsIDString();
  
  /// 
  /// DisplayModifiedEvent is generated when display node parameters is changed
  enum
    {
      DisplayModifiedEvent = 20000
    };

  /// 
  /// default file name for freesurfer labels
  vtkGetStringMacro(LabelsFileName);
  vtkSetStringMacro(LabelsFileName);
  
  virtual int GetNumberOfColors();
  virtual bool GetColor(int entry, double* color);

protected:
  vtkMRMLFreeSurferProceduralColorNode();
  ~vtkMRMLFreeSurferProceduralColorNode();
  vtkMRMLFreeSurferProceduralColorNode(const vtkMRMLFreeSurferProceduralColorNode&);
  void operator=(const vtkMRMLFreeSurferProceduralColorNode&);

  /// 
  /// Set values in the names vector from the colour rgba entries in the colour
  /// table
  bool SetNameFromColor(int index);

  /// 
  /// a lookup table tailored with FreeSurfer colours, constructed according to Type
  vtkFSLookupTable *LookupTable;

  char *LabelsFileName;
};

#endif
