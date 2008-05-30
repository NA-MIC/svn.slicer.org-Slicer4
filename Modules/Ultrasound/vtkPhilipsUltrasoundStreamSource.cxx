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

void vtkPhilipsUltrasoundStreamSource::Reconnect()
{
    this->StopStreaming();
    this->StartStreaming();
}  

void vtkPhilipsUltrasoundStreamSource::StartStreaming()
{
    if (this->StreamManager.Start((char*)this->IPAddress.c_str()))
        this->SetConnected();
}

void vtkPhilipsUltrasoundStreamSource::StopStreaming()
{
    if (this->IsConnected())
    {
        this->StreamManager.Stop();
        this->SetDisconnected();
    }
}

bool _cdecl vtkPhilipsUltrasoundStreamSource::StreamingCallback(_int64 frame_index, SClient3DArray *echo_data, SClient3DArray *color_data)
{
    if (me != NULL)
    {
        unsigned char *checkVolumePt1, *checkVolumePt2, *checkVolumePt3;
        checkVolumePt1 = echo_data->pData + (70) + (40)*echo_data->width + ( 30)*echo_data->height*echo_data->width;
        checkVolumePt2 = echo_data->pData + (70) + (40)*echo_data->width + (100)*echo_data->height*echo_data->width;
        checkVolumePt3 = echo_data->pData + (70) + (40)*echo_data->width + (170)*echo_data->height*echo_data->width;
        bool volumeIsNotUsable = ( *checkVolumePt1==255 && *checkVolumePt1==255 && *checkVolumePt3==255 );
        if (volumeIsNotUsable)
            return false;

        me->SetDataInHiddenBuffer(echo_data->pData, echo_data->width, echo_data->height, echo_data->depth);
        return true;
    }
    return false;
}

void vtkPhilipsUltrasoundStreamSource::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}


