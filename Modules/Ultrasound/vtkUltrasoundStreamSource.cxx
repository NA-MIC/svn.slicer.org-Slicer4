#include "vtkUltrasoundStreamSource.h"

#include "vtkImageData.h"
#include "vtkMutexLock.h"


vtkCxxRevisionMacro(vtkUltrasoundStreamSource, "$Revision 1.0 $");

//#include "vtkPhilipsUltrasoundStreamSource.h"
vtkUltrasoundStreamSource* vtkUltrasoundStreamSource::New()
{
    return NULL; //vtkPhilipsUltrasoundStreamSource::New();
}

vtkUltrasoundStreamSource::vtkUltrasoundStreamSource(void)
{
    this->CurrentBuffer = 0;
    this->b_Connected = false;
    this->b_DataReady = false;
    for (int i=0; i<2; i++)
    {
        this->ImageBuffers[i] = vtkImageData::New();
        this->ImageBuffers[i]->SetScalarTypeToUnsignedChar();
        this->ImageBuffers[i]->SetNumberOfScalarComponents(1);
        this->ImageBuffers[i]->SetExtent(0,9, 0,9, 0,9);
        this->ImageBuffers[i]->SetSpacing(1.0f, 1.0f, 0.74f);
        this->ImageBuffers[i]->AllocateScalars();
        memset(this->ImageBuffers[i]->GetScalarPointer(), 0, 10*10*10*this->ImageBuffers[i]->GetScalarSize());
    }
    this->Mutex = vtkSimpleMutexLock::New();
}

vtkUltrasoundStreamSource::~vtkUltrasoundStreamSource(void)
{
    this->ImageBuffers[0]->Delete();
    this->ImageBuffers[1]->Delete();
    this->Mutex->Delete();
}

void vtkUltrasoundStreamSource::SwapBuffers()
{
    this->CurrentBuffer = (this->CurrentBuffer == 0) ? 1 : 0;
}

vtkImageData* vtkUltrasoundStreamSource::GetData() 
{ 
    return this->ImageBuffers[this->CurrentBuffer]; 
}
    
vtkImageData* vtkUltrasoundStreamSource::GetDataInHiddenBuffer() 
{ 
    return this->ImageBuffers[(this->CurrentBuffer == 0)? 1 : 0]; 
}

void vtkUltrasoundStreamSource::FetchImageData(vtkImageData* data)
{
    this->Mutex->Lock();
    if (this->b_DataReady == true)
        this->SwapBuffers();
    data->DeepCopy(this->GetData());
    this->b_DataReady = false;
    this->Mutex->Unlock();
}

void vtkUltrasoundStreamSource::SetDataInHiddenBuffer(vtkImageData* data)
{
    Mutex->Lock();
    vtkImageData* buffer = this->GetDataInHiddenBuffer();
    buffer->DeepCopy(data);
    this->b_DataReady = true;
    Mutex->Unlock();
}

void vtkUltrasoundStreamSource::SetDataInHiddenBuffer(unsigned char* data, int width, int height, int depth)
{
    Mutex->Lock();
    vtkImageData* buffer = this->GetDataInHiddenBuffer();
    int* old_dimensions = buffer->GetDimensions();
    if (old_dimensions[0] != width || old_dimensions[1] != height || old_dimensions[2] != depth)
    {
        buffer->SetExtent(0, width-1, 0, height-1, 0, depth-1);
        buffer->AllocateScalars();
    }
    int mem_size = width*height*depth;
    memcpy( buffer->GetScalarPointer(), data, mem_size);
    this->b_DataReady = true;
    Mutex->Unlock();
}

void vtkUltrasoundStreamSource::SetConnected()
{
    this->b_Connected = true;
    this->InvokeEvent(vtkUltrasoundStreamSource::ConnectionEstablished);
}

void vtkUltrasoundStreamSource::SetDisconnected()
{
    this->b_Connected = false;
    this->InvokeEvent(vtkUltrasoundStreamSource::ConnectionClosed);
}


void vtkUltrasoundStreamSource::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}


