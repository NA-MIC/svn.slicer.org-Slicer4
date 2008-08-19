#ifndef __vtkDataIOManagerLogic_h
#define __vtkDataIOManagerLogic_h

#include "vtkObject.h"
#include "vtkUnsignedLongArray.h"
#include "vtkSmartPointer.h"
#include "vtkIntArray.h"
#include "vtkMRML.h"

#include "vtkSlicerModuleLogic.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerTask.h"

#include <list>
#include <string>

#include "vtkDataTransfer.h"
#include "vtkDataIOManager.h"
#include "vtkMRMLNode.h"


#ifndef vtkObjectPointer
#define vtkObjectPointer(xx) (reinterpret_cast <vtkObject **>( (xx) ))
#endif


class VTK_SLICER_BASE_LOGIC_EXPORT vtkDataIOManagerLogic : public vtkSlicerModuleLogic
{
  public:
  
  // The Usual vtk class functions
  static vtkDataIOManagerLogic *New();
  vtkTypeRevisionMacro(vtkDataIOManagerLogic,vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent);
  
  vtkGetObjectMacro ( DataIOManager, vtkDataIOManager );
  virtual void SetAndObserveDataIOManager ( vtkDataIOManager *);
  
  virtual void ProcessMRMLEvents( vtkObject *caller, unsigned long event, void *calldata );

  // Description:
  // Methods that Queues the read
  virtual int QueueRead ( vtkMRMLNode *node );

  // Description:
  // Method that queues the write 
  virtual int QueueWrite ( vtkMRMLNode *node );

  // Description:
  // The method that executes the data transfer in another thread
  virtual void ApplyTransfer(void *clientdata);

  // Description
  // Communicates progress back to the DataIOManager
  static void ProgressCallback ( void * );

  // Description:
  // Convenience method that goes through vtkDataIOManager
  // to create a new DataTransfer object.
  virtual void AddNewDataTransfer ( vtkDataTransfer *transfer, vtkMRMLNode *node );
  virtual void CancelDataTransfer ( vtkDataTransfer *transfer );
  virtual void ClearCache();
  virtual void DeleteDataTransferFromCache ( vtkDataTransfer *transfer);

 private:
  vtkDataIOManager *DataIOManager;
  
 protected:
  vtkDataIOManagerLogic();
  virtual ~vtkDataIOManagerLogic();
  vtkDataIOManagerLogic(const vtkDataIOManagerLogic&);
  void operator=(const vtkDataIOManagerLogic&);

};

#endif

