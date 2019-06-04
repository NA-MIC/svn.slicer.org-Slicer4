/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Kitware Inc.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Simon Drouin, Brigham and Women's
  Hospital, Boston, MA.

==============================================================================*/
/// vtkMRMLShaderPropertyNode - MRML node to represent custom volume shader
/// variables and replacement code

#ifndef __vtkMRMLShaderPropertyNode_h
#define __vtkMRMLShaderPropertyNode_h

// VolumeRendering includes
#include "vtkSlicerVolumeRenderingModuleMRMLExport.h"

// MRML includes
#include "vtkMRMLStorableNode.h"

// VTK includes
class vtkShaderProperty;
class vtkUniforms;

/// \brief vtkMRMLShaderPropertyNode volume shader custom code and
/// custom uniform variables defined by users or specialized rendering
/// modules.
class VTK_SLICER_VOLUMERENDERING_MODULE_MRML_EXPORT vtkMRMLShaderPropertyNode
  : public vtkMRMLStorableNode
{
public:

  /// Create a new vtkMRMLShaderPropertyNode
  static vtkMRMLShaderPropertyNode *New();
  vtkTypeMacro(vtkMRMLShaderPropertyNode,vtkMRMLStorableNode);
  void PrintSelf(ostream& os, vtkIndent indent) VTK_OVERRIDE;

  /// Don't change its scalarOpacity, gradientOpacity or color on the volume property
  /// but use the methods below. It wouldn't observe them.
  vtkGetObjectMacro(ShaderProperty, vtkShaderProperty);

  /// Get the list of user-defined uniform variables.
  vtkUniforms * GetVertexUniforms();
  vtkUniforms * GetFragmentUniforms();
  vtkUniforms * GetGeometryUniforms();

  //--------------------------------------------------------------------------
  // MRMLNode methods
  //--------------------------------------------------------------------------
  virtual vtkMRMLNode* CreateNodeInstance() VTK_OVERRIDE;

  /// Set node attributes
  virtual void ReadXMLAttributes( const char** atts) VTK_OVERRIDE;

  /// Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent) VTK_OVERRIDE;

  /// Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node) VTK_OVERRIDE;

  /// Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() VTK_OVERRIDE {return "ShaderProperty";}

  /// Reimplemented for internal reasons.
  virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData) VTK_OVERRIDE;

  /// Create default storage node or NULL if does not have one
  virtual vtkMRMLStorageNode* CreateDefaultStorageNode() VTK_OVERRIDE;

  /// \sa vtkMRMLStorableNode::GetModifiedSinceRead()
  virtual bool GetModifiedSinceRead() VTK_OVERRIDE;

protected:
  vtkMRMLShaderPropertyNode(void);
  ~vtkMRMLShaderPropertyNode(void) override;

protected:
  /// Events observed on the transfer functions
  vtkIntArray* ObservedEvents;

  /// Main parameters for visualization
  vtkShaderProperty* ShaderProperty;

private:
  vtkMRMLShaderPropertyNode(const vtkMRMLShaderPropertyNode&) = delete;
  void operator=(const vtkMRMLShaderPropertyNode&) = delete;

};

#endif
