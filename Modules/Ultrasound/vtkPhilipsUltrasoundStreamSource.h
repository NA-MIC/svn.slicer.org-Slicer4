#ifndef __vtkPhilipsUltrasoundStreamSource_h
#define __vtkPhilipsUltrasoundStreamSource_h

#include "vtkUltrasoundStreamSource.h"
#include "vtkUltrasoundModule.h"
#include <string>

#include "StreamMgr.h"

class VTK_ULTRASOUNDMODULE_EXPORT vtkPhilipsUltrasoundStreamSource : public vtkUltrasoundStreamSource
{
public:
    static vtkPhilipsUltrasoundStreamSource *New();
    vtkTypeRevisionMacro(vtkPhilipsUltrasoundStreamSource, vtkUltrasoundStreamSource);
    
    void PrintSelf(ostream& os, vtkIndent indent);

    virtual void Reconnect();
    virtual void StartStreaming();
    virtual void StopStreaming();

    //BTX
    virtual void SetSourceAddress(const std::string& sourceAddress) { this->SetIPAddress(sourceAddress); }
    void SetIPAddress(const std::string& ip_address);
    //ETX
protected:
    vtkPhilipsUltrasoundStreamSource(void);
    ~vtkPhilipsUltrasoundStreamSource(void);
    
    //BTX
    static bool _cdecl StreamingCallback(_int64 frame_index, SClient3DArray *echo_data, SClient3DArray *color_data);
    static vtkPhilipsUltrasoundStreamSource* me;

    CStreamMgr StreamManager;
    std::string IPAddress;
    //ETX
private:
    vtkPhilipsUltrasoundStreamSource(const vtkPhilipsUltrasoundStreamSource&); //Not implemented
    void operator=(const vtkPhilipsUltrasoundStreamSource&);  
};

#endif /* __vtkPhilipsUltrasoundStreamSource_h */
