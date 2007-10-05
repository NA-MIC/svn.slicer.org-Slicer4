#include "vtkLabelMapColorTransferFunction.h"
#include "vtkObjectFactory.h"

vtkLabelMapColorTransferFunction* vtkLabelMapColorTransferFunction::New(void)
{
     // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkLabelMapColorTransferFunction");
  if(ret)
    {
      return (vtkLabelMapColorTransferFunction*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkLabelMapColorTransferFunction;
}
vtkLabelMapColorTransferFunction::vtkLabelMapColorTransferFunction(void)
{
}

vtkLabelMapColorTransferFunction::~vtkLabelMapColorTransferFunction(void)
{
}
void vtkLabelMapColorTransferFunction::Init(vtkMRMLScalarVolumeNode *node)
{
    //test if inputdata is valid
    if(node==NULL)
    {
        vtkErrorMacro("No node");
        return;
    }
    if(node->GetVolumeDisplayNode()==NULL)
    {
        vtkErrorMacro("No Volume Display Node");
        return;
    }
    if(node->GetVolumeDisplayNode()->GetColorNode()==NULL)
    {
        vtkErrorMacro("No Color Node");
        return;
    }
     if (node->GetLabelMap()==0)
    {
        vtkErrorMacro("this is not a labelMap");
        return;
    }
    vtkLookupTable *lookup=node->GetVolumeDisplayNode()->GetColorNode()->GetLookupTable();
    if(lookup==NULL)
    {
        vtkErrorMacro("No LookupTable");
        return;
    }
    this->AdjustRange(lookup->GetTableRange());
    double color[3];

    for (int i=(int)lookup->GetTableRange()[0];i<lookup->GetTableRange()[1];i++)
    {
        lookup->GetColor(i,color);
        this->AddRGBPoint(i,color[0],color[1],color[2]);
        this->AddRGBPoint(i+.9999,color[0],color[1],color[2]);

    }
}
