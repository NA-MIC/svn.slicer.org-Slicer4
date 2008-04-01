// .NAME vtkMRMLUltrasoundNode - MRML node to represent ultrasound information
// .SECTION Description
// This node is especially used to store visualization parameter sets for volume rendering


#ifndef __vtkMRMLUltrasoundNode_h
#define __vtkMRMLUltrasoundNode_h

#include "vtkMRMLNode.h"
#include "vtkUltrasoundModule.h"

#include <string>
#include "vtkstd/vector"

class VTK_ULTRASOUNDMODULE_EXPORT vtkMRMLUltrasoundNode : public vtkMRMLNode
{
public:
    //--------------------------------------------------------------------------
    // OWN methods
    //--------------------------------------------------------------------------
    
    //ETX
    // Description:
    // Create a new vtkMRMLUltrasoundNode
    static vtkMRMLUltrasoundNode *New();
    vtkTypeMacro(vtkMRMLUltrasoundNode,vtkMRMLNode);
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
    // Copy the node's attributes to this object
    virtual void Copy(vtkMRMLNode *node);

    //Description;
    //Copy only the paramterset (like Volume Propertys, Piecewiesefunctions etc. as deep copy,but no references etc.)
    void CopyParameterset(vtkMRMLNode *node);

    // Description:
    // Get node XML tag name (like Volume, Model)
    virtual const char* GetNodeTagName() {return "Ultrasound";};

    // Description:
    // 
    virtual void UpdateScene(vtkMRMLScene *scene);
    virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData);

    //BTX
    void AddReference(std::string id);
    bool HasReference(std::string id);
    void RemoveReference(std::string id);
    //ETX
protected:
    // Description:
    // Use ::New() to get a new instance.
    vtkMRMLUltrasoundNode(void);
    // Description:
    // Use ->Delete() to delete object
    ~vtkMRMLUltrasoundNode(void);

    //BTX
    // Description:
    // References to vtkMRMLScalarVolumeNodes
    vtkstd::vector<std::string> References;
    //ETX

private:
    // Description:
    // Caution: Not implemented
    vtkMRMLUltrasoundNode(const vtkMRMLUltrasoundNode&);//Not implemented
    void operator=(const vtkMRMLUltrasoundNode&);// Not implmented
};

#endif
