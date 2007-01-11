

#ifndef IGTOPENTRACKERSTREAM_H
#define IGTOPENTRACKERSTREAM_H


#include "vtkMatrix4x4.h"
#include "vtkIGTDataStream.h"

#include "OpenTracker/OpenTracker.h"
#include "OpenTracker/common/CallbackModule.h"
using namespace ot;


class vtkIGTOpenTrackerStream : public vtkIGTDataStream
{
public:


    static vtkIGTOpenTrackerStream *New();
      vtkTypeMacro(vtkIGTOpenTrackerStream,vtkIGTDataStream);
      void PrintSelf(ostream& os, vtkIndent indent);


  vtkIGTOpenTrackerStream ( );

  Init(int device_type, char* configfile);


  virtual ~vtkIGTOpenTrackerStream ( );




protected:

  
  
 private:

  vtkIGTMatrixState* matrixstate;
  //vtkIGTImageState* imagestate;

  int  Initialize_Opentracker(char* configfile);
  static void CallbackF(const Node&,const Event &event, void ata);
  Context *context;
  quaternion2xyz(float* orientation, float *normal,float *transnormal);
};

#endif // IGTOPENTRACKERSTREAM_H
