#ifndef __vtkUltrasoundScanPlane_h
#define __vtkUltrasoundScanPlane_h

#include "vtkObject.h"
#include "vtkUltrasoundModule.h"

class vtkMatrix4x4;
class vtkRenderer;
class vtkPlaneSource;
class vtkCylinderSource;
class vtkPolyDataMapper;
class vtkActor;

class VTK_ULTRASOUNDMODULE_EXPORT vtkUltrasoundScanPlane : public vtkObject
{
public:
    static vtkUltrasoundScanPlane *New();
    vtkTypeMacro(vtkUltrasoundScanPlane, vtkObject);

    void SetRenderer(vtkRenderer* renderer);
    void SetScale(float x, float y);
    void SetUserMatrix(vtkMatrix4x4* Transform);

protected:
    void CollectRevisions(ostream& sos);
    vtkUltrasoundScanPlane();
    ~vtkUltrasoundScanPlane();

    vtkUltrasoundScanPlane(const vtkUltrasoundScanPlane& );            //not defined
    vtkUltrasoundScanPlane& operator=(const vtkUltrasoundScanPlane& ); //not defined

private:
    float                Scale[2];

    vtkPlaneSource*        Plane;
    vtkCylinderSource*    Cylinders[4];

    vtkPolyDataMapper*  Mappers[5];
    vtkActor*            Actors[5];
};


#endif /* __vtkUltrasoundScanPlane_h */
