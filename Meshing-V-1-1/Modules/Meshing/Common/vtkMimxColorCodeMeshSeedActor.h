/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkMimxColorCodeMeshSeedActor.h,v $
Language:  C++
Date:      $Date: 2008/07/06 20:30:13 $
Version:   $Revision: 1.4 $

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
// .NAME vtkMimxColorCodeMeshSeedActor - a 3D cube with face labels
// .SECTION Description
// vtkMimxColorCodeMeshSeedActor is a hybrid 3D actor used to represent an anatomical
// orientation marker in a scene.  The class consists of a 3D unit cube centered
// on the origin with each face labelled in correspondance to a particular
// coordinate direction.  For example, with Cartesian directions, the user
// defined text labels could be: +X, -X, +Y, -Y, +Z, -Z, while for anatomical
// directions: A, P, L, R, S, I.  Text is automatically centered on each cube
// face and is not restriceted to single characters. In addition to or in
// replace of a solid text label representation, the outline edges of the labels
// can be displayed.  The individual properties of the cube, face labels
// and text outlines can be manipulated as can their visibility.

// .SECTION Caveats
// vtkMimxColorCodeMeshSeedActor is primarily intended for use with
// vtkOrientationMarkerWidget. The cube face text is generated by vtkVectorText
// and therefore the font attributes are restricted.

// .SECTION See Also
// vtkAxesActor vtkOrientationMarkerWidget vtkVectorText

#ifndef __vtkMimxColorCodeMeshSeedActor_h
#define __vtkMimxColorCodeMeshSeedActor_h

#include "vtkProp3D.h"

class vtkActor;
class vtkPropCollection;
class vtkScalarBarActor;
class vtkUnstructuredGrid;

class vtkMimxColorCodeMeshSeedActor : public vtkProp3D
{
public:
  static vtkMimxColorCodeMeshSeedActor *New();
  vtkTypeRevisionMacro(vtkMimxColorCodeMeshSeedActor,vtkProp3D);
  void PrintSelf(ostream& os, vtkIndent indent);
  vtkGetObjectMacro(MeshSeedActor, vtkActor);
  vtkGetObjectMacro(ScalarBarActor, vtkScalarBarActor);
  vtkSetMacro(LineWidth, int);
  // Description:
  // For some exporters and other other operations we must be
  // able to collect all the actors or volumes. These methods
  // are used in that process.
  virtual void GetActors(vtkPropCollection *);

  // Description:
  // Support the standard render methods.
  virtual int RenderOpaqueGeometry(vtkViewport *viewport);
  virtual int RenderTranslucentPolygonalGeometry(vtkViewport *viewport);

  // Description:
  // Does this prop have some translucent polygonal geometry?
  virtual int HasTranslucentPolygonalGeometry();
  
  // Description:
  // Shallow copy of an axes actor. Overloads the virtual vtkProp method.
  void ShallowCopy(vtkProp *prop);

  // Description:
  // Release any graphics resources that are being consumed by this actor.
  // The parameter window could be used to determine which graphic
  // resources to release.
  void ReleaseGraphicsResources(vtkWindow *);

  // Description:
  // Get the bounds for this Actor as (Xmin,Xmax,Ymin,Ymax,Zmin,Zmax). (The
  // method GetBounds(double bounds[6]) is available from the superclass.)
  void GetBounds(double bounds[6]);
  double *GetBounds();

  // Description:
  // Get the actors mtime plus consider its properties and texture if set.
  unsigned long int GetMTime();

  // Description:
  // Return the mtime of anything that would cause the rendered image to
  // appear differently. Usually this involves checking the mtime of the
  // prop plus anything else it depends on such as properties, textures
  // etc.
  virtual unsigned long GetRedrawMTime();
  void SetInput(vtkUnstructuredGrid *MeshSeedInput);
  
  void SetLabelTextColor(double color[3]);

protected:
  vtkMimxColorCodeMeshSeedActor();
  ~vtkMimxColorCodeMeshSeedActor();

  void UpdateProps();
  vtkUnstructuredGrid *Input;
  int MinMeshSeed;
  int MaxMeshSeed;
  vtkActor *MeshSeedActor;
  vtkScalarBarActor *ScalarBarActor;
  int LineWidth;
private:
  vtkMimxColorCodeMeshSeedActor(const vtkMimxColorCodeMeshSeedActor&);  // Not implemented.
  void operator=(const vtkMimxColorCodeMeshSeedActor&);  // Not implemented.
  
  double TextColor[3];
};

#endif

