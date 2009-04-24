/*=========================================================================

  Program:   Slicer3

  Copyright (c) Brigham and Women's Hospital (BWH) All Rights Reserved.

  See License.txt or http://www.slicer.org/copyright/copyright.txt for details.

==========================================================================*/
// .NAME vtkMRMLParser - Parse XML scene file
// .SECTION Description


#ifndef __vtkMRMLParser_h
#define __vtkMRMLParser_h

#include <stack> 

#include "vtkObjectFactory.h"
#include "vtkXMLParser.h"

#include "vtkMRML.h"
#include "vtkMRMLScene.h"

class VTK_MRML_EXPORT vtkMRMLParser : public vtkXMLParser
{
public:
  static vtkMRMLParser *New();
  vtkTypeMacro(vtkMRMLParser,vtkCollection);
  void PrintSelf(ostream& os, vtkIndent indent){}
  
  vtkMRMLScene* GetMRMLScene() {return this->MRMLScene;};
  void SetMRMLScene(vtkMRMLScene* scene) {this->MRMLScene = scene;};

  vtkCollection* GetNodeCollection() {return this->NodeCollection;};
  void SetNodeCollection(vtkCollection* scene) {this->NodeCollection = scene;};
  
protected:
  vtkMRMLParser() : MRMLScene(NULL),NodeCollection(NULL){};
  ~vtkMRMLParser() {};
  vtkMRMLParser(const vtkMRMLParser&);
  void operator=(const vtkMRMLParser&);
  
  virtual void StartElement(const char* name, const char** atts);
  virtual void EndElement (const char *name);

private:
  vtkMRMLScene* MRMLScene;
  vtkCollection* NodeCollection;
//BTX
  std::stack< vtkMRMLNode *> NodeStack;
//ETX
};

#endif
