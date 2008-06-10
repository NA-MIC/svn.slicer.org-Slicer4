#ifndef __vtkUltrasoundStreamSource_h
#define __vtkUltrasoundStreamSource_h

#include "vtkObject.h"
#include "vtkUltrasoundModule.h"

class vtkImageData;
class vtkSimpleMutexLock;
#include <string>

class VTK_ULTRASOUNDMODULE_EXPORT vtkUltrasoundStreamSource : public vtkObject
{
public:
    //BTX
    enum
    {
        ConnectionEstablished = 12300,
        ConnectionClosed,
    };
    //ETX

    static vtkUltrasoundStreamSource *New();
    vtkTypeRevisionMacro(vtkUltrasoundStreamSource, vtkObject);
    
    void PrintSelf(ostream& os, vtkIndent indent);


    void SwapBuffers();
    vtkImageData* GetData();
    void FetchImageData(vtkImageData* data);

    bool    IsConnected() const { return b_Connected; } 

    void SetSpacings(float x, float y, float z);

    virtual void Reconnect() = 0;
    virtual void StartStreaming() = 0;
    virtual void StopStreaming() = 0;
    //BTX
    virtual void SetSourceAddress(const std::string& sourceAddress) = 0;
    //ETX

protected:
    // Description:
    // Use ::New() to get a new instance.
    vtkUltrasoundStreamSource(void);
    // Description:
    // Use ->Delete() to delete object
    ~vtkUltrasoundStreamSource(void);
    
    void SetDataInHiddenBuffer(vtkImageData* data);
    void SetDataInHiddenBuffer(unsigned char* data, int width, int height, int depth);
    vtkImageData* GetDataInHiddenBuffer();

    void                    SetConnected();
    void                    SetDisconnected();
    
protected:
    vtkImageData*           ImageBuffers[2];
    int                     CurrentBuffer;
    vtkSimpleMutexLock*     Mutex;

    bool                    b_Connected;
    bool                    b_DataReady;

private:
    vtkUltrasoundStreamSource(const vtkUltrasoundStreamSource&); //Not implemented
    void operator=(const vtkUltrasoundStreamSource&);  
};

#endif /* __vtkUltrasoundStreamSource_h */

