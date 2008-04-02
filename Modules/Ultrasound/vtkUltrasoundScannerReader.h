#ifndef __vtkUltrasoundScannerReader_h
#define __vtkUltrasoundScannerReader_h

#include "vtkObject.h"
#include "vtkUltrasoundModule.h"

class vtkMultiThreader;
class vtkImageData;
class vtkCallbackCommand;

class VTK_ULTRASOUNDMODULE_EXPORT vtkUltrasoundScannerReader : public vtkObject
{
public:
    static vtkUltrasoundScannerReader *New();
    vtkTypeRevisionMacro(vtkUltrasoundScannerReader, vtkObject);
    
    void PrintSelf(ostream& os, vtkIndent indent);

    void SwapBuffers();
    vtkImageData* GetData() { return this->ImageBuffers[this->CurrentBuffer]; }
    vtkImageData* GetDataInHiddenBuffer() { return this->ImageBuffers[(this->CurrentBuffer == 0)? 1 : 0]; }

    vtkCallbackCommand* DataUpdated;

    void StartScanning();
    void StopScanning();

protected:

    static VTK_THREAD_RETURN_TYPE StartThread(void* data);
    void UpdateData();

    // Description:
    // Use ::New() to get a new instance.
    vtkUltrasoundScannerReader(void);
    // Description:
    // Use ->Delete() to delete object
    ~vtkUltrasoundScannerReader(void);

    
    vtkImageData* ImageBuffers[2];
    int CurrentBuffer;


    vtkMultiThreader* Thread;
    bool ThreadRunning;

private:
    // Description:
    // Caution: Not implemented
    vtkUltrasoundScannerReader(const vtkUltrasoundScannerReader&); //Not implemented
    void operator=(const vtkUltrasoundScannerReader&);             // Not implmented
};

#endif /* __vtkUltrasoundScannerReader_h */
