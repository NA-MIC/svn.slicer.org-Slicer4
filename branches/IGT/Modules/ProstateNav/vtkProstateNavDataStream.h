#ifndef __vtkProstateNavDataStream_h
#define __vtkProstateNavDataStream_h

#include "vtkIGTWin32Header.h" 
#include "vtkIGTDataStream.h"

#include "vtkMatrix4x4.h"

#include "vtkIGTOpenTrackerStream.h"
#include "vtkIGTMessageAttributeSet.h"


class VTK_IGT_EXPORT vtkProstateNavDataStream : public vtkIGTOpenTrackerStream
{
public:

  static vtkProstateNavDataStream *New();
  vtkTypeRevisionMacro(vtkProstateNavDataStream,vtkIGTOpenTrackerStream);
  void PrintSelf(ostream& os, vtkIndent indent);

  //Description:
  //Constructor
  vtkProstateNavDataStream();

  //Description:
  //Destructor
  virtual ~vtkProstateNavDataStream();
  
  void Init(const char *configFile);
  static void callbackF(const Node&, const Event &event, void *data);
  static void GenericCallback(const Node &node, const Event &event, void *data);
  
  // Register callback functions which were called from GenericCallback().
  // AddCallback() should be called before Init();
  //BTX
  void AddCallbacks();
  //ETX

  /*
  void StopPulling();
  void PullRealTime();
  */

  static void OnRecieveMessageFromRobot(vtkIGTMessageAttributeSet* data, void* arg);
  static void OnRecieveMessageFromScanner(vtkIGTMessageAttributeSet* data, void* arg);
  
private:

  //Context *context;
  
  //void CloseConnection();
  vtkIGTMessageAttributeSet* AttrSetRobot;
  vtkIGTMessageAttributeSet* AttrSetScanner;

  vtkMatrix4x4* NeedleMatrix;

};

#endif // __vtkProstateNavDataStream_h




