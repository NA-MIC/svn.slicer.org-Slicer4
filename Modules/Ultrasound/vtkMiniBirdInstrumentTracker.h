#ifndef __vtkMiniBirdInstrumentTracker_h
#define __vtkMiniBirdInstrumentTracker_h

#include "vtkObject.h"
#include "vtkUltrasoundModule.h"

class vtkMatrix4x4;

class VTK_ULTRASOUNDMODULE_EXPORT vtkMiniBirdInstrumentTracker : public vtkObject
{
public:
    static vtkMiniBirdInstrumentTracker *New();
    vtkTypeMacro(vtkMiniBirdInstrumentTracker, vtkObject);

    vtkMatrix4x4*    GetTransform() const { return this->Transform; }

    float            GetPosX() const { return this->PosX; }
    float            GetPosY() const    { return this->PosY; }
    float            GetPosZ() const    { return this->PosZ; }

    float            GetPhi() const { return this->Phi; }
    float            GetTheta() const { return this->Theta; }
    float            GetRoll() const { return this->Roll; }
    void            CalcInstrumentPos();

protected:
    void CollectRevisions(ostream& sos);
    vtkMiniBirdInstrumentTracker();
    virtual ~vtkMiniBirdInstrumentTracker();

private:
    const int m_NumberOfDevices;

    vtkMatrix4x4*    Transform;
    float            PosX;
    float            PosY;
    float            PosZ;
    float            Phi;
    float            Theta;
    float            Roll;
};

#endif /* __vtkMiniBirdInstrumentTracker_h */
