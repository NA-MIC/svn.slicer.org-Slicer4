#include "vtkUltrasoundScannerReader.h"
#include "vtkObjectFactory.h"

// Temporary
#include "vtkMultiThreader.h"
//#include "vtkCallbackCommand.h"

#if defined (__WIN32__) && ! defined (_POSIX_VERSION)
#include "windows.h"
#endif


//Temporary

vtkCxxRevisionMacro(vtkUltrasoundScannerReader, "$Revision 1.0 $");
vtkStandardNewMacro(vtkUltrasoundScannerReader);


vtkUltrasoundScannerReader::vtkUltrasoundScannerReader(void)
{
    //this->DataUpdated = vtkCallbackCommand::New();
    this->Thread = NULL;
    this->ThreadRunning = false;
    this->ThreadAlive = false;
}

vtkUltrasoundScannerReader::~vtkUltrasoundScannerReader(void)
{
    //this->DataUpdated->Delete();
    this->StopStreaming();
}



#include "vtkstd/vector"
#include "vtkImageReader.h"

void vtkUltrasoundScannerReader::SetFileName(const std::string& file_name)
{
    this->FileName = file_name;
}

VTK_THREAD_RETURN_TYPE vtkUltrasoundScannerReader::StartThread(void* data)
{
    vtkMultiThreader::ThreadInfo* info =  (vtkMultiThreader::ThreadInfo*)data;
    ((vtkUltrasoundScannerReader*)info->UserData)->UpdateData();

    return VTK_THREAD_RETURN_VALUE;
}

void vtkUltrasoundScannerReader::UpdateData()
{
    if (this->ThreadRunning)
        return;
    this->ThreadRunning = true;
    this->ThreadAlive = true;

    std::cout << "Loading " << std::flush;
    int j = 92;
    unsigned int i;
    for (i = 0; i < 50; i ++)
    {
        ImageReaders.push_back(vtkImageReader::New());
        ImageReaders[i]->SetDataScalarTypeToUnsignedChar();
        ImageReaders[i]->SetNumberOfScalarComponents(1);
        ImageReaders[i]->SetDataExtent(0, 79, 0, 79, 0, 207);
        ImageReaders[i]->SetFileDimensionality(3);
        ImageReaders[i]->SetDataSpacing(0.64f, 0.64f, 0.403f);
        ImageReaders[i]->SetHeaderSize(i * 80 * 80 * 208 * sizeof(unsigned char));

        ImageReaders[i]->SetFileName(this->FileName.c_str());
        ImageReaders[i]->Update();
        std::cout << "." << std::flush;

        if (ThreadRunning == false)
            break;
    }
    std::cout << "done" <<  std::endl;

    this->FrameNumber = 0;
    
    this->SetConnected();

    do
    {
      // SLEEP
    } while(this->ThreadRunning);

    while (!ImageReaders.empty())
    {
        vtkImageReader* reader = ImageReaders.back();
        ImageReaders.pop_back();
        reader->Delete();
    }
    for (i = 0; i < ImageReaders.size(); i++)
        ImageReaders[i]->Delete();

    this->ThreadAlive = false;
}

void vtkUltrasoundScannerReader::RequestRead()
{
    if (ImageReaders.empty())
        return;

    if (++FrameNumber >= ImageReaders.size())
            FrameNumber = 0;
    this->SetDataInHiddenBuffer(ImageReaders[FrameNumber]->GetOutput());
}

void vtkUltrasoundScannerReader::Reconnect()
{
    this->StopStreaming();
    this->StartStreaming();
}

void vtkUltrasoundScannerReader::StartStreaming()
{
    if (this->Thread == NULL)
    {
        this->Thread = vtkMultiThreader::New();
        //this->Thread->SetSingleMethod(vtkUltrasoundScannerReader::StartThread, this);
    }
    this->Thread->SpawnThread(vtkUltrasoundScannerReader::StartThread, (void*)this);
}

void vtkUltrasoundScannerReader::StopStreaming()
{
    if (this->Thread != NULL)
    {
        this->ThreadRunning = false;
        while(this->ThreadAlive)
            ;//Sleep(10);
        this->SetDisconnected();
    }
}

void vtkUltrasoundScannerReader::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
