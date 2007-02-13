

#ifndef IGTOPENTRACKERSTREAM_H
#define IGTOPENTRACKERSTREAM_H


#include "vtkIGTWin32Header.h" 
#include "vtkMatrix4x4.h"
#include "vtkIGTDataStream.h"
#include "vtkIGTMatrixState.h"


#include "OpenTracker/OpenTracker.h"
#include "OpenTracker/common/CallbackModule.h"
using namespace ot;


class VTK_IGT_EXPORT vtkIGTOpenTrackerStream : public vtkIGTDataStream
{
 public:
  
  
  static vtkIGTOpenTrackerStream *New();
  vtkTypeRevisionMacro(vtkIGTOpenTrackerStream,vtkIGTDataStream);
  void PrintSelf(ostream& os, vtkIndent indent);
  
  
  vtkIGTOpenTrackerStream ( );
  void Init(char* configfile);
  virtual ~vtkIGTOpenTrackerStream ( );
  

  static void callbackF(const Node&,const Event &event, void *data);
  
 protected:

  //vtkIGTImageState* imagestate;
  
  Context *context;
  void quaternion2xyz(float* orientation, float *normal,float *transnormal);
};

#endif // IGTOPENTRACKERSTREAM_H
