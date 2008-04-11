#include "vtkPhilipsUltrasoundStreamSource.h"
#include "vtkObjectFactory.h"

vtkCxxRevisionMacro(vtkPhilipsUltrasoundStreamSource, "$Revision 1.0 $");

vtkPhilipsUltrasoundStreamSource* vtkPhilipsUltrasoundStreamSource::me = NULL;

vtkPhilipsUltrasoundStreamSource* vtkPhilipsUltrasoundStreamSource::New()
{
    if (me == NULL)
        me = new vtkPhilipsUltrasoundStreamSource();
    return me;
}
 
vtkPhilipsUltrasoundStreamSource::vtkPhilipsUltrasoundStreamSource(void)
{
    this->IPAddress = "";
    this->StreamManager.RegisterCallback(vtkPhilipsUltrasoundStreamSource::StreamingCallback);
    this->StreamManager.Initialize();
}

vtkPhilipsUltrasoundStreamSource::~vtkPhilipsUltrasoundStreamSource(void)
{
    this->StreamManager.Shutdown();
}

void vtkPhilipsUltrasoundStreamSource::SetIPAddress(const std::string& ip_address)
{
    this->IPAddress = ip_address;
}

void vtkPhillipsUltrasoundStreamSource::Reconnect()
{
    this->StopStreaming();
    this->StartStreaming();
}  

void vtkPhilipsUltrasoundStreamSource::StartStreaming()
{
    this->StreamManager.Start((char*)this->IPAddress.c_str());
}

void vtkPhilipsUltrasoundStreamSource::StopStreaming()
{
    this->StreamManager.Stop();
}

bool _cdecl vtkPhilipsUltrasoundStreamSource::StreamingCallback(_int64 frame_index, SClient3DArray *echo_data, SClient3DArray *color_data)
{
    if (me != NULL)
        me->SetDataInHiddenBuffer(echo_data->pData, echo_data->width, echo_data->height, echo_data->depth);
    return true;
}

void vtkPhilipsUltrasoundStreamSource::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}


