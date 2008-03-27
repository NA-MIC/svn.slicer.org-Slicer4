#include "vtkObject.h"
#include "vtkObjectFactory.h"

#include "vtkSlicerGradientEditorLogic.h"

#include "vtkMRMLDiffusionWeightedVolumeNode.h"
#include "vtkMRMLNRRDStorageNode.h"
#include "vtkDoubleArray.h"
#include "vtkNRRDReader.h"
#include <string.h>
#include "vtkTimerLog.h"

vtkCxxRevisionMacro(vtkSlicerGradientEditorLogic, "$Revision: 1.9.12.1 $");
vtkStandardNewMacro(vtkSlicerGradientEditorLogic);

//---------------------------------------------------------------------------
vtkSlicerGradientEditorLogic::vtkSlicerGradientEditorLogic(void)
  {
  this->ActiveVolumeNode = NULL;
  this->StackPosition = 0;
  this->UndoFlag = 0;
  }

//---------------------------------------------------------------------------
vtkSlicerGradientEditorLogic::~vtkSlicerGradientEditorLogic(void)
  {
  if (this->ActiveVolumeNode)
    {
    this->ActiveVolumeNode->Delete();
    this->ActiveVolumeNode = NULL;
    }
  if (!this->UndoRedoStack.empty())
    {
    for(unsigned int i = 0; i<this->UndoRedoStack.size(); i++)
      {
      this->UndoRedoStack.at(i)->Delete();
      this->UndoRedoStack.at(i) = NULL;
      }
    }
  this->UndoRedoStack.clear();
  this->StackPosition = 0;
  this->UndoFlag = 0;
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorLogic::PrintSelf ( ostream& os, vtkIndent indent )
  {
  this->vtkObject::PrintSelf ( os, indent );
  os << indent << "vtkSlicerGradientEditorLogic: " << this->GetClassName ( ) << "\n";
  }

//---------------------------------------------------------------------------
int vtkSlicerGradientEditorLogic::AddGradients (const char* filename, int numberOfGradients, vtkDoubleArray *newBValue, 
                                                vtkDoubleArray *newGradients)
  {
  // format the filename
  std::string fileString(filename);
  for (unsigned int i = 0; i < fileString.length(); i++)
    {
    if (fileString[i] == '\\') fileString[i] = '/';
    }

  // Instanciation of the I/O mechanism
  vtkMRMLNRRDStorageNode *storageNode = vtkMRMLNRRDStorageNode::New();
  vtkMRMLDiffusionWeightedVolumeNode *dwiNode = vtkMRMLDiffusionWeightedVolumeNode::New();
  dwiNode->SetBValues(newBValue);
  dwiNode->SetDiffusionGradients(newGradients);
  storageNode->SetFileName(fileString.c_str());

  if (!storageNode->ReadData(dwiNode))
    {
    storageNode->Delete();
    dwiNode->Delete();

    //check if txt file
    std::string suffix(".txt");
    std::string::size_type pos = fileString.find(suffix);
    if(pos == std::string::npos)
      {
      //no txt file or valid nhdr
      vtkWarningMacro("no valid file");
      return 0;
      }

    ifstream file;
    file.open(fileString.c_str(), ios::in);

    if(file.good())
      {
      std::stringstream content;
      file.seekg(0L, ios::beg);
      while (!file.eof())
        {
        char c;
        file.get(c);
        content<<c;
        }
      vtkWarningMacro("\n"<<content.str().c_str());
      return this->ParseGradients(content.str().c_str(), numberOfGradients, newBValue, newGradients);
      }
    return 0;
    }
  storageNode->Delete();
  dwiNode->Delete();
  return 1;
  }

//---------------------------------------------------------------------------
int vtkSlicerGradientEditorLogic::StringToDouble(const std::string &s, double &result)
  {
  std::stringstream stream (s);
  if(stream >> result)
    {
    if(stream.eof()) return 1;
    }
  return 0;
  }

//---------------------------------------------------------------------------
int vtkSlicerGradientEditorLogic::ParseGradients(const char *oldGradients, int numberOfGradients,
                                                 vtkDoubleArray *newBValues, vtkDoubleArray *newGradients)
  {
  if (oldGradients == NULL || oldGradients == "")
    {
    vtkErrorMacro(<< this->GetClassName() << ": oldGradients is NULL");
    return 0;
    }

  // read in current gradients 
  std::stringstream grad;
  grad << oldGradients;

  // save all values in a vector
  std::vector<double> vec;
  while(!grad.eof())
    {
    std::string dummy = "";
    double newValue = -1;
    grad >> dummy;
    if(StringToDouble(dummy, newValue))
      {
      vec.push_back(newValue);
      }      
    }

  // exit if too many or to less values are input
  if(vec.size() != numberOfGradients*3+1)
    {
    vtkWarningMacro("given values "<<vec.size()<<", needed "<<numberOfGradients*3+1);
    return 0;
    }

  vtkDoubleArray *factor = vtkDoubleArray::New();
  newGradients->SetNumberOfComponents(3);
  newGradients->SetNumberOfTuples(numberOfGradients);
  newBValues->SetNumberOfTuples(numberOfGradients);

  // set gradients and factor values
  for(unsigned int j = 1; j < vec.size(); j=j+3)
    {
    for(unsigned int i=j; i<j+3;i++)
      {
      newGradients->SetValue(i-1,vec[i]);
      }
    factor->InsertNextValue(sqrt(vec[j]*vec[j]+vec[j+1]*vec[j+1]+vec[j+2]*vec[j+2]));
    }

  // get range
  double range[2];
  factor->GetRange(range);

  // set BValues
  for (int i=0; i<numberOfGradients;i++)
    {
    newBValues->SetValue(i,vec[0]*factor->GetValue(i)/range[1]);
    }

  factor->Delete();
  return 1;
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorLogic::SetActiveVolumeNode(vtkMRMLDiffusionWeightedVolumeNode *node)
  {
  vtkSetMRMLNodeMacro(this->ActiveVolumeNode, node); //set activeVolumeNode
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorLogic::SaveStateForUndoRedo()
  {
  //new node comes in, delete nodes in stack that are no longer reachable 
  //meaning: all nodes after current StackPosition
  if(!this->UndoRedoStack.empty() && this->StackPosition != this->UndoRedoStack.size())
    {
    while(this->StackPosition != this->UndoRedoStack.size())
      {
      this->UndoRedoStack.pop_back();
      }
    }
  if(!this->UndoFlag)
    {
    //create node and save it in stack
    vtkMRMLDiffusionWeightedVolumeNode *nodeToSave = vtkMRMLDiffusionWeightedVolumeNode::New();
    nodeToSave->Copy(this->ActiveVolumeNode);
    this->UndoRedoStack.push_back(nodeToSave);
    this->StackPosition = this->UndoRedoStack.size(); //update StackPosition

    }
  this->UndoFlag = 0;
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorLogic::UpdateActiveVolumeNode(vtkMRMLDiffusionWeightedVolumeNode *node)
  {
  vtkTimerLog *timer = vtkTimerLog::New();
  timer->StartTimer();

  vtkMatrix4x4 *m = vtkMatrix4x4::New();
  node->GetMeasurementFrameMatrix(m);
  this->ActiveVolumeNode->SetMeasurementFrameMatrix(m);

  timer->StopTimer();
  vtkWarningMacro("time1: "<<timer->GetElapsedTime());
  timer->StartTimer();

  this->ActiveVolumeNode->SetDiffusionGradients(node->GetDiffusionGradients());

  timer->StopTimer();
  vtkWarningMacro("time2: "<<timer->GetElapsedTime());
  timer->StartTimer();

  this->ActiveVolumeNode->SetBValues(node->GetBValues());


  timer->StopTimer();
  vtkWarningMacro("time3: "<<timer->GetElapsedTime());
  timer->Delete();

  m->Delete();
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorLogic::Restore()
  {
  if(!this->UndoRedoStack.empty())
    {
    vtkMRMLDiffusionWeightedVolumeNode *node = this->UndoRedoStack.at(0);
    this->UpdateActiveVolumeNode(node);
    this->UndoRedoStack.clear();
    this->StackPosition = 0;
    vtkErrorMacro("size: "<<this->UndoRedoStack.size()<<" pos: "<<this->StackPosition);
    }
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorLogic::Undo()
  {
  //the first time you click undo, save the parmeters
  vtkErrorMacro("size: "<<this->UndoRedoStack.size()<<" pos: "<<this->StackPosition<<", in undo: "<<this->UndoFlag);
  if(this->StackPosition == this->UndoRedoStack.size() && !this->UndoFlag)
    {
    this->SaveStateForUndoRedo();
    this->UndoFlag = 1;
    }

  if(!this->UndoRedoStack.empty() && this->IsUndoable())
    {
    this->StackPosition--;
    vtkMRMLDiffusionWeightedVolumeNode *node = this->UndoRedoStack.at(this->StackPosition-1);
    this->UpdateActiveVolumeNode(node);
    vtkErrorMacro("size: "<<this->UndoRedoStack.size()<<" pos: "<<this->StackPosition);
    }
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorLogic::Redo()
  {
  if(!this->UndoRedoStack.empty() && this->IsRedoable())
    {
    this->StackPosition++;
    vtkMRMLDiffusionWeightedVolumeNode *node;
    node = this->UndoRedoStack.at(this->StackPosition-1);
    this->UpdateActiveVolumeNode(node);
    vtkErrorMacro("size: "<<this->UndoRedoStack.size()<<" pos: "<<this->StackPosition);
    }
  }

//---------------------------------------------------------------------------
int vtkSlicerGradientEditorLogic::IsUndoable()
  {
  if((this->UndoRedoStack.size()+1 > this->StackPosition && this->StackPosition > 1) || !this->UndoFlag) return 1;
  else return 0;
  }

//---------------------------------------------------------------------------
int vtkSlicerGradientEditorLogic::IsRedoable()
  {
  if(0 < this->StackPosition && this->StackPosition < this->UndoRedoStack.size()) return 1;
  else return 0;
  }


