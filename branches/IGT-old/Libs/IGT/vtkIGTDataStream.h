
#ifndef IGTDATASTREAM_H
#define IGTDATASTREAM_H

#include <string>
#include <vector>

#include "vtkIGTWin32Header.h" 
#include "vtkObject.h"

#include "vtkMatrix4x4.h"
#include "vtkTransform.h"


#define IGT_MATRIX_STREAM 0
#define IGT_IMAGE_STREAM 1

class VTK_IGT_EXPORT vtkIGTDataStream : public vtkObject
{
public:

    // Constructors/Destructors
    //  Magic lines for vtk and Slicer
    static vtkIGTDataStream *New();
    vtkTypeRevisionMacro(vtkIGTDataStream,vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent);


    vtkIGTDataStream ();
    ~vtkIGTDataStream ();

    // Description:
    // Set the rate at which to stream data.  You must provide a value
    // in milleseconds to specify how often a new data record will be
    // acquired.
    vtkSetMacro(Speed,int);

    // Description:
    // Position multiplication factor.  This can be used, for
    // instance, to convert from units of millimetres to units of metres.
    vtkSetMacro(MultiFactor,float);

    // Description:
    // Use this method to start and stop the streaming of locator
    // data.
    vtkSetMacro(Tracking,int);

    // Description:
    // Set the registration matrix.  If this is set, the raw
    // locator information will be transformed by the RegMatrix
    // before it is stored in the LocatorMatrix.
    vtkSetObjectMacro(RegMatrix,vtkMatrix4x4);
    vtkGetObjectMacro(RegMatrix,vtkMatrix4x4);

    // Description:
    // A matrix describing the current location.  The first column
    // is the position, the second column is the tool normal (the
    // the tool axis for a pointer, or the Y-axis in the tool-oriented
    // coordinate system), and the third column is third column is
    // the tool transnormal (the X-axis in the tool-oriented
    // coordinate system). 
    vtkGetObjectMacro(LocatorMatrix,vtkMatrix4x4);

    // Description:
    // The LocatorNormalTransform is a vtkTransform constructed
    // from the information in the LocatorMatrix.
    vtkGetObjectMacro(LocatorNormalTransform,vtkTransform);

    // Description:
    // Internal method that will be called after Tracking has been
    // set to false.
    virtual void StopPulling() {};

    // Description:
    // Internal method that will be called each time that the stream
    // needs to poll for fresh data.
    virtual void PullRealTime() {};

    // Description:
    // Internal method to set the LocatorNormalTransform from the
    // information in the LocatorMatrix.
    virtual void SetLocatorTransforms();

    // Description:
    // Internal method that is called every few milliseconds.  The
    // rate depending on what Speed is set to.  When it is called,
    // it calls either PullRealTime() or StopPulling() depending on
    // whether Tracking is true or false.
    virtual void ProcessTimerEvents();

protected:

    int Speed;
    int Tracking;
    float MultiFactor;

    vtkMatrix4x4 *LocatorMatrix;
    vtkMatrix4x4 *RegMatrix;
    vtkTransform *LocatorNormalTransform;

    // Description:
    // A utility function to convert a quaternion, stored in xyzw
    // order, to a normal and transnormal.  The normal and transnormal
    // are, respectively, the 2nd and 1st column of a 3x3 rotation
    // matrix that describes the same rotation as the quaternion.
    void QuaternionToXYZ(float *orientation, float *normal, float *transnormal); 
    // Description:
    // Apply the RegMatrix to the locator data described by a
    // position, normal, and transnormal.
    void ApplyTransform(float *position, float *norm, float *transnorm);
 };

#endif // IGTDATASTREAM_H
