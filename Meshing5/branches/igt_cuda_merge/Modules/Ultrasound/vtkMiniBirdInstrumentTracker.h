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
    void             CalcMatrices();
    void             CalcInstrumentPos();

    void             SetToolAdjustment(float x, float y, float z) { ToolAdjustments[0]=x; ToolAdjustments[1]=y; ToolAdjustments[2]=z; }
    void             SetProbeAdjustment(float x, float y, float z) { ProbeAdjustments[0]=x;ProbeAdjustments[0]=y; ProbeAdjustments[0]=z; }
    void             SetTranslationScale(float scale) { this->TranslationScale = scale; }

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

    float             ToolAdjustments[3];
    float             ProbeAdjustments[3];
    float             TranslationScale;
};

#endif /* __vtkMiniBirdInstrumentTracker_h */
