#ifndef __vtkUltrasoundScannerReader_h
#define __vtkUltrasoundScannerReader_h

#include "vtkUltrasoundStreamSource.h"
#include "vtkUltrasoundModule.h"

#include <string>
class vtkMultiThreader;
//class vtkCallbackCommand;

class VTK_ULTRASOUNDMODULE_EXPORT vtkUltrasoundScannerReader : public vtkUltrasoundStreamSource
{
public:
    static vtkUltrasoundScannerReader *New();
    vtkTypeRevisionMacro(vtkUltrasoundScannerReader, vtkObject);
    
    void PrintSelf(ostream& os, vtkIndent indent);

    //vtkCallbackCommand* DataUpdated;

    virtual void Reconnect();
    virtual void StartStreaming();
    virtual void StopStreaming();
    
    //BTX
    virtual void SetSourceAddress(const std::string& sourceAddress) { this->SetFileName(sourceAddress); }
    void SetFileName(const std::string& file_name);
    //ETX
protected:

    static VTK_THREAD_RETURN_TYPE StartThread(void* data);
    void UpdateData();

    // Description:
    // Use ::New() to get a new instance.
    vtkUltrasoundScannerReader(void);
    // Description:
    // Use ->Delete() to delete object
    ~vtkUltrasoundScannerReader(void);

protected:    
    vtkMultiThreader*       Thread;
    bool                    ThreadAlive;
    bool                    ThreadRunning;

    //BTX
    std::string             FileName;
    //ETX
private:
    // Description:
    // Caution: Not implemented
    vtkUltrasoundScannerReader(const vtkUltrasoundScannerReader&); //Not implemented
    void operator=(const vtkUltrasoundScannerReader&);             // Not implmented
};

#endif /* __vtkUltrasoundScannerReader_h */
