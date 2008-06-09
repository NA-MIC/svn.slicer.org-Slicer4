#include "vtkUltrasoundScanPlane.h"
#include "vtkObjectFactory.h"

#include "vtkMatrix4x4.h"
#include "vtkActor.h"
#include "vtkPlaneSource.h"
#include "vtkCylinderSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkRenderer.h"


vtkCxxRevisionMacro(vtkUltrasoundScanPlane, "$Revision 1.0 $");
vtkStandardNewMacro(vtkUltrasoundScanPlane);


vtkUltrasoundScanPlane::vtkUltrasoundScanPlane()
{
    this->Scale[0] = 1.0; this->Scale[1] = 1.0;
    unsigned int i;
    this->Plane = vtkPlaneSource::New();

    for (i = 0; i < 5; i++)
    {
        this->Mappers[i] = vtkPolyDataMapper::New();
        this->Actors[i] = vtkActor::New();

        this->Actors[i]->SetMapper(this->Mappers[i]);
    }

    for (i = 0; i < 4; i++)
    {
        this->Cylinders[i] = vtkCylinderSource::New();
        this->Cylinders[i]->SetRadius(.05);
        this->Cylinders[i]->SetHeight(1.0);
        this->Mappers[i]->SetInput(this->Cylinders[i]->GetOutput());
    }

    this->Actors[0]->RotateZ(90);
    this->Actors[1]->RotateZ(90);
    this->Cylinders[0]->SetCenter(0.5, 0, 0);
    this->Cylinders[1]->SetCenter(-0.5,0, 0);
    this->Cylinders[2]->SetCenter(0.5, 0, 0);
    this->Cylinders[3]->SetCenter(-0.5, 0, 0);
    

    this->Mappers[4]->SetInput(this->Plane->GetOutput());
//    this->Actors[4]->SetOpacity(.2);
}

vtkUltrasoundScanPlane::~vtkUltrasoundScanPlane()
{
    unsigned int i;
    this->Plane->Delete();
    for (i = 0; i < 4; i++)
        this->Cylinders[i]->Delete();
    for (i = 0; i < 5; i++)
    {
        this->Actors[i]->SetMapper(NULL);
        this->Actors[i]->Delete();
        this->Mappers[i]->Delete();
    }
}

void vtkUltrasoundScanPlane::SetRenderer(vtkRenderer* renderer)
{
    for (unsigned i = 0; i < 5; i++)
        renderer->AddActor(this->Actors[i]);
}

void vtkUltrasoundScanPlane::SetScale(float x, float y)
{
    this->Scale[0] = x; 
    this->Scale[1] = y;
}

void vtkUltrasoundScanPlane::SetUserMatrix(vtkMatrix4x4* Transform)
{
                vtkMatrix4x4* mat = vtkMatrix4x4::New();
            vtkMatrix4x4* mul = vtkMatrix4x4::New();

            mul->Identity();
            mul->SetElement(0,0, 50);
            mul->SetElement(1,1, 50);
            vtkMatrix4x4::Multiply4x4(Transform, mul, mat);


    for(unsigned int i = 0; i < 5; i++)
    {
        this->Actors[i]->SetUserMatrix(mat);
    }
    mat->Delete();
    mul->Delete();
}
