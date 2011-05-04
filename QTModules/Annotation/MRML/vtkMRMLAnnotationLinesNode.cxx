#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "vtkObjectFactory.h"
#include "vtkMRMLAnnotationLinesStorageNode.h"
#include "vtkMRMLAnnotationLinesNode.h"
#include "vtkMRMLAnnotationLineDisplayNode.h"
#include "vtkMRMLAnnotationPointDisplayNode.h"
#include "vtkMRMLAnnotationTextDisplayNode.h"
#include "vtkBitArray.h"
#include "vtkCellArray.h"
#include "vtkDataSetAttributes.h"
#include "vtkPointData.h"
#include "vtkStringArray.h"


//------------------------------------------------------------------------------
vtkMRMLAnnotationLinesNode* vtkMRMLAnnotationLinesNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLAnnotationLinesNode");
  if(ret)
    {
    return (vtkMRMLAnnotationLinesNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLAnnotationLinesNode;
}

//-----------------------------------------------------------------------------
vtkMRMLNode* vtkMRMLAnnotationLinesNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLAnnotationLinesNode");
  if(ret)
    {
    return (vtkMRMLAnnotationLinesNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLAnnotationLinesNode;
}


//----------------------------------------------------------------------------
vtkMRMLAnnotationLinesNode::vtkMRMLAnnotationLinesNode()
{
  this->InitializeLinesFlag = 0;
}

//----------------------------------------------------------------------------
vtkMRMLAnnotationLinesNode::~vtkMRMLAnnotationLinesNode()
{
}

//----------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);
  
  vtkIndent indent(nIndent);

  if (this->PolyData &&  this->GetNumberOfLines()) 
    { 
      int n = this->GetNumberOfLines();

      vtkCellArray *lines = this->PolyData->GetLines();  
      lines->InitTraversal();
      of << indent << "linePtsID=\"" ;
      for (int i = 0; i < n; i++ ) 
    {
      vtkIdType npts = 0;
      vtkIdType *pts = NULL;
      lines->GetNextCell(npts, pts);
      for (int j= 0; j < npts ; j++)
        {
          of << pts[j];
          if (j<  npts -1)
        {
          of << " " ;
        }
        }
      if (i < n-1) 
        {
          of << "|" ;
        }
    }
      of << "\"";


      for (int j = NUM_CP_ATTRIBUTE_TYPES ; j < NUM_LINE_ATTRIBUTE_TYPES; j ++) 
    {
      of << indent << " " << this->GetAttributeTypesEnumAsString(j)<<"=\"";
      for (int i = 0; i < n -1 ; i++ ) 
        {
          of << this->GetAnnotationAttribute(i,j) << " " ;
        }
      if (n) 
        {
          of << this->GetAnnotationAttribute(n-1,j) ;
        }

      of << "\"";
    }  
    }
  else 
    {
      of << indent << "linePtsID=\"\"" ;
      for (int j = NUM_CP_ATTRIBUTE_TYPES ; j < NUM_LINE_ATTRIBUTE_TYPES; j ++) 
    {
      of << indent << " " << this->GetAttributeTypesEnumAsString(j) << "=\"\"";
    }
    }
}


//----------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::ReadXMLAttributes(const char** atts)
{
  // cout << "vtkMRMLAnnotationLinesNode::ReadXMLAttributes start"<< endl;

  int disabledModify = this->StartModify();

  Superclass::ReadXMLAttributes(atts);

  while (*atts != NULL) 
    {
    const char* attName = *(atts++);
    std::string attValue(*(atts++));
    if (!strcmp(attName, "linePtsID"))       
      {
    std::string valStr(attValue);
    std::replace(valStr.begin(), valStr.end(), '|', ' ');

    std::stringstream ss;
    ss << valStr;
    int d;
    std::vector<int> tmpVec;
    while (ss >> d)
        {
      tmpVec.push_back(d);
        }

    for (int i = 0; i < int(tmpVec.size()); i += 2) 
      {
        this->AddLine(tmpVec[i],tmpVec[i + 1],0,0);
      }
       
      }
    else
      {
    int j = NUM_CP_ATTRIBUTE_TYPES;
    while (j <  NUM_LINE_ATTRIBUTE_TYPES) 
      {     
        if (!strcmp(attName, this->GetAttributeTypesEnumAsString(j))) 
          {
        std::stringstream ss;
        ss << attValue;
        double value;
        vtkIdType id = 0;
        while (ss >> value)
          {
            this->SetAnnotationAttribute(id,j, value);
            id ++;
          }
        j = NUM_LINE_ATTRIBUTE_TYPES;
          }
        j++;
      }
      }
    }
  this->EndModify(disabledModify);
}

//-----------------------------------------------------------
void vtkMRMLAnnotationLinesNode::UpdateScene(vtkMRMLScene *scene)
{
  Superclass::UpdateScene(scene);

  // Nothing to do at this point  bc vtkMRMLAnnotationLineDisplayNode is subclass of vtkMRMLModelDisplayNode 
  // => will be taken care of by vtkMRMLModelDisplayNode  

}


//----------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
}

//---------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event, 
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);

  // Not necessary bc vtkMRMLAnnotationLineDisplayNode is subclass of vtkMRMLModelDisplayNode 
  // => will be taken care of  in vtkMRMLModelNode
}

//----------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::PrintAnnotationInfo(ostream& os, vtkIndent indent, int titleFlag)
{
  if (titleFlag) 
    {
      os <<indent << "vtkMRMLAnnotationLinesNode: Annotation Summary";
      if (this->GetName()) 
    {
      os << " of " << this->GetName();
    }
      os << endl;
    }

  Superclass::PrintAnnotationInfo(os, indent, 0);


  if (this->PolyData &&  this->GetNumberOfLines()) 
    { 

      int n = this->GetNumberOfLines();
      vtkCellArray *lines = this->PolyData->GetLines();  
      lines->InitTraversal();
      os << indent << "linePtsID: " ;
      for (int i = 0; i < n; i++ ) 
    {
      vtkIdType npts;
      vtkIdType *pts = NULL;
      lines->GetNextCell(npts, pts);
      if (!pts)
        {
          os << "Not Defined ";
        } 
      else
        {
          for (int j= 0; j < npts ; j++)
        {
          os << pts[j];
          if (j<  npts -1)
            {
              os << " " ;
            }
        }
        }
      if (i < n-1) 
        {
          os << "|" ;
        }
    }
      os << endl;

      for (int j = NUM_CP_ATTRIBUTE_TYPES ; j < NUM_LINE_ATTRIBUTE_TYPES; j ++) 
    {
      os << indent << this->GetAttributeTypesEnumAsString(j) <<": ";
      for (int i = 0; i < n; i++ ) 
        {
          os << this->GetAnnotationAttribute(i,j) << " " ;
        }
      os << endl;
    }
    } 
  else 
    {
      os << indent << "linePtsID: None" << endl;
      for (int j = NUM_CP_ATTRIBUTE_TYPES ; j < NUM_LINE_ATTRIBUTE_TYPES; j ++) 
    {
      os << indent << this->GetAttributeTypesEnumAsString(j) <<": None" << endl;
    }
    }
}

//----------------------------------------------------------------------------
vtkMRMLAnnotationLineDisplayNode* vtkMRMLAnnotationLinesNode::GetAnnotationLineDisplayNode()
{
  int nnodes = this->GetNumberOfDisplayNodes();
  for (int n=0; n<nnodes; n++)
    {
    vtkMRMLAnnotationLineDisplayNode * node = vtkMRMLAnnotationLineDisplayNode::SafeDownCast(this->GetNthDisplayNode(n));
    
    if (node && node->IsA("vtkMRMLAnnotationLineDisplayNode")) 
      {
    return node;
      }
    }
  return 0;
}


//---------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::CreateAnnotationLineDisplayNode()
{
  vtkMRMLAnnotationLineDisplayNode *node = this->GetAnnotationLineDisplayNode();
  if (node) return;
  if (!this->GetScene()) 
    {
      vtkErrorMacro("vtkMRMLAnnotationLinesNode::CreateAnnotationLineDisplayNode AnnotationLine: No scene defined" ) ;
      return;
    }

  node = vtkMRMLAnnotationLineDisplayNode::New();
  node->SetScene(this->GetScene());
  this->GetScene()->AddNode(node);
  node->Delete();
  this->AddAndObserveDisplayNodeID(node->GetID());
  // This assumes I want to display the poly data, which I do not 
  // node->SetPolyData(this->GetPolyData());
}

//---------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::ResetAnnotations()
{
  Superclass::ResetAnnotations();
  this->ResetLines();
}

//---------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::CreatePolyData() 
{
  Superclass::CreatePolyData(); 

  if (!this->GetPolyData()->GetLines() || !this->InitializeLinesFlag)
    {
      // Remember vtkCellArray *polyLines = this->PolyData->GetLines() is never null as it is a static variable !
      vtkCellArray *polyLines = vtkCellArray::New();
      this->PolyData->SetLines(polyLines);
      polyLines->Delete();
      this->InitializeLinesFlag = 1;     
    }
}

//---------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::ResetLines()
{

  this->CreatePolyData();
  this->PolyData->GetLines()->Reset();
  this->ResetLinesAttributesAll();
}

//---------------------------------------------------------------------------
int vtkMRMLAnnotationLinesNode::GetNumberOfLines()
{
  if (!this->PolyData || !this->PolyData->GetLines()) {
    this->ResetLines();
  }
  // have to do it this way bc  this->PolyData->GetLines() is never null before it is initilized ! 
  return this->PolyData->GetNumberOfLines();
}


//---------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::DeleteLine(int id)
{
  if (!this->PolyData || !this->PolyData->GetLines()) 
    {
      this->ResetLines();
    }

  vtkCellArray *lines = this->PolyData->GetLines();  
  int n = this->GetNumberOfLines();
  if (id< 0 || id > n-1) 
    {
      vtkErrorMacro("AnnotationLine " << this->GetName() << " id is out of range !"); 
      return ;
    }


  // create event in hearder when deleted 
  lines->InitTraversal();
  // cellLine->SetTraversalLocation(id);

  vtkIdType *cPts = NULL;
  vtkIdType cNpts = 0;
  vtkIdType *nPts = NULL;
  vtkIdType nNpts = 0;
  for (int i = 0; i <= id; i++ ) 
    {
      lines->GetNextCell(cNpts, cPts);
    }
  for (int i = id; i < n-1; i++ ) 
    {
      lines->GetNextCell(nNpts, nPts);
      if (nNpts != 2 || cNpts != 2)
    {
      vtkErrorMacro("Error in cell structure " << cNpts  << " " << nNpts);
      return;
    }
      cPts[0]=nPts[0]; cPts[1]=nPts[1];
      cPts=nPts;
    }
  lines->SetNumberOfCells(n-1);

  for (int j = NUM_CP_ATTRIBUTE_TYPES ; j < NUM_LINE_ATTRIBUTE_TYPES; j ++) 
    {
      vtkBitArray *dataArray = dynamic_cast <vtkBitArray *> (this->GetAnnotationAttributes(j));
      if (!dataArray || dataArray->GetSize() != n) 
        {
          vtkErrorMacro("Annotation " << this->GetName() << " Attribute " << this->GetAttributeTypesEnumAsString(j) << " is out of sync with number of lines" );
        } 
      else 
    {
      this->DeleteAttribute(id,j); 
    }
    }
}

//---------------------------------------------------------------------------
int vtkMRMLAnnotationLinesNode::GetEndPointsId(vtkIdType id, vtkIdType ctrlPtsID[2])
{

  if (!this->PolyData || !this->GetNumberOfLines()) 
    {
      vtkErrorMacro("Annotation: " << this->GetName() << " PolyData or  PolyData->GetLines() is NULL" ); 
      return 0;
    }
  
  if (id >= this->GetNumberOfLines())
    {
      vtkErrorMacro("Annotation: GetEndPointsId: id is out of range");
      return 0;
    }
 
  vtkCellArray* lines = this->PolyData->GetLines();
  lines->InitTraversal();

  vtkIdType npts = 0;
  vtkIdType *pts = NULL;
  
  for (int i = 0; i < id; i++ ) 
    {
      lines->GetNextCell(npts, pts);
    }

  lines->GetNextCell(npts, pts);
  
  if (npts != 2) 
    {
      vtkErrorMacro("Annotation: GetEndPointsId: line with id is not correctly defined");
      return 0;
    }
  ctrlPtsID[0] = pts[0];
  ctrlPtsID[1] = pts[1];
  return 1; 
}



//---------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::ResetLinesAttributesAll() {
  for (int j = NUM_CP_ATTRIBUTE_TYPES ; j < NUM_LINE_ATTRIBUTE_TYPES; j ++) {
    this->ResetAttributes(j);
  }
} 

//-------------------------------------------------------------------------
const char *vtkMRMLAnnotationLinesNode::GetAttributeTypesEnumAsString(int val) 
{
  if (val < vtkMRMLAnnotationControlPointsNode::NUM_CP_ATTRIBUTE_TYPES) {
    return vtkMRMLAnnotationControlPointsNode::GetAttributeTypesEnumAsString(val);
  }
  if (val == LINE_SELECTED) return "lineSelected";
  if (val == LINE_VISIBLE) return "lineVisible";
  return "(unknown)";
};

//-------------------------------------------------------------------------
vtkMRMLStorageNode* vtkMRMLAnnotationLinesNode::CreateDefaultStorageNode()
{
  return vtkMRMLStorageNode::SafeDownCast(vtkMRMLAnnotationLinesStorageNode::New());
}


//---------------------------------------------------------------------------
int  vtkMRMLAnnotationLinesNode::SetLine(int id, int ctrlPtIdStart, int ctrlPtIdEnd,int selectedFlag, int visibleFlag)
{
  if (id < 0) 
  {
    vtkErrorMacro("Invalid ID");
    return 0;
  }

  // Create if not there
  if (!this->PolyData || !this->PolyData->GetPoints()) {
    vtkErrorMacro("No Control Points defined");
    return 0;
  }


  vtkPoints *points = this->PolyData->GetPoints(); 
  if (!points->GetPoint(ctrlPtIdStart) || !points->GetPoint(ctrlPtIdEnd)) 
    {
      vtkErrorMacro("At least one control point does not exists");
      return 0;  
    }

  // Define line 
  vtkIdType lineIdList[2] = {ctrlPtIdStart, ctrlPtIdEnd} ;

  // Enter into Array, by creating cell array, traversing to the right location, and entering it
  if (! this->GetNumberOfLines()) 
    {
      this->CreatePolyData();
    }

  vtkCellArray* cellLine = this->PolyData->GetLines();
 
  if (cellLine->GetNumberOfCells() <= id ) 
    {
      if (cellLine->GetNumberOfCells()) 
    {
      cellLine->SetTraversalLocation(cellLine->GetNumberOfCells()-1);
    }

      int addCells = id - cellLine->GetNumberOfCells();
      for (int i=0; i< addCells; i++)
    {
      cellLine->InsertNextCell(0,0);
    }  
      cellLine->InsertNextCell(2,lineIdList);
    }
  else 
    {
      cellLine->InitTraversal();
      cellLine->ReplaceCell(id,2,lineIdList);
    }

  for (int j = NUM_CP_ATTRIBUTE_TYPES ; j < NUM_LINE_ATTRIBUTE_TYPES; j ++) 
    {
      this->SetAttributeSize(j,this->GetNumberOfLines());
    }

  this->SetAnnotationAttribute(id, LINE_SELECTED, selectedFlag);
  this->SetAnnotationAttribute(id, LINE_VISIBLE, visibleFlag);
  return 1;
}

//---------------------------------------------------------------------------
int vtkMRMLAnnotationLinesNode::AddLine(int ctrlPtIdStart, int ctrlPtIdEnd,int selectedFlag, int visibleFlag)
{
  // Create if not there
  if (!this->PolyData || !this->GetNumberOfLines()) 
    {
      this->ResetLines();
    }

  // Look if line is already included
  int n = this->GetNumberOfLines();
  int id = n;
  for (int i =0 ; i < n ; i++) 
    {
      vtkIdType endPts[2];
      this->GetEndPointsId(i, endPts);
      if ((endPts[0] == ctrlPtIdStart) && (endPts[1] == ctrlPtIdEnd))
    {
      id = i;
      break;
    }
    } 

  if (this->SetLine(id,  ctrlPtIdStart, ctrlPtIdEnd,selectedFlag, visibleFlag)) 
    {
      return id;
    }
  return -1;
}

//---------------------------------------------------------------------------
int vtkMRMLAnnotationLinesNode::SetControlPoint(int id, double newControl[3],int selectedFlag, int visibleFlag)
{
  int result = vtkMRMLAnnotationControlPointsNode::SetControlPoint(id,newControl,selectedFlag,visibleFlag);
  // cout << "vtkMRMLAnnotationLinesNode::SetControlPoint "<< id << " " << result << endl;
  if (result && this->GetAnnotationLineDisplayNode())
    {
      // This is needed bc otherwise the bounding box of ruler is not correctly updated
      this->GetAnnotationTextDisplayNode()->SetVisibility(1);
      this->GetAnnotationTextDisplayNode()->SetVisibility(0);
      this->GetAnnotationPointDisplayNode()->SetVisibility(1);
      this->GetAnnotationPointDisplayNode()->SetVisibility(0);
      this->GetAnnotationLineDisplayNode()->SetVisibility(1);
      this->GetAnnotationLineDisplayNode()->SetVisibility(0);
    }
  return result;
}

//---------------------------------------------------------------------------
void vtkMRMLAnnotationLinesNode::Initialize(vtkMRMLScene* mrmlScene)
{
   if (!mrmlScene)
   {
     vtkErrorMacro("Scene was null!")
     return;
   }
 
   // we need to disable the modified event which would get fired when we set the new displayNode
   this->DisableModifiedEventOn();
   if (this->GetScene()!=mrmlScene)
     {
     this->SetScene(mrmlScene);
     }
   this->CreateAnnotationLineDisplayNode();
   this->DisableModifiedEventOff();

   Superclass::Initialize(mrmlScene);
}
