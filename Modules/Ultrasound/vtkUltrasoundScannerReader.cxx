#include "vtkUltrasoundScannerReader.h"
#include "vtkObjectFactory.h"


// Temporary
#include "vtkMultiThreader.h"
#include "vtkImageData.h"
#include "vtkCallbackCommand.h"


//Temporary

vtkCxxRevisionMacro(vtkUltrasoundScannerReader, "$Revision 1.0 $");
vtkStandardNewMacro(vtkUltrasoundScannerReader);


vtkUltrasoundScannerReader::vtkUltrasoundScannerReader(void)
{
    this->CurrentBuffer = 0;
    this->ImageBuffers[0] = vtkImageData::New();
    this->ImageBuffers[1] = vtkImageData::New();

    this->DataUpdated = vtkCallbackCommand::New();

    this->Thread = NULL;
    this->ThreadRunning = false;

}

vtkUltrasoundScannerReader::~vtkUltrasoundScannerReader(void)
{
    this->ImageBuffers[0]->Delete();
    this->ImageBuffers[1]->Delete();

    this->DataUpdated->Delete();
}


void vtkUltrasoundScannerReader::SwapBuffers()
{
    this->CurrentBuffer = (this->CurrentBuffer == 0) ? 1 : 0;
}


VTK_THREAD_RETURN_TYPE vtkUltrasoundScannerReader::StartThread(void* data)
{
    ((vtkUltrasoundScannerReader*) data)->UpdateData();

    return VTK_THREAD_RETURN_VALUE;
}

#include "vtkstd/vector"
#include "vtkImageReader.h"

void vtkUltrasoundScannerReader::UpdateData()
{
    this->ThreadRunning = true;

    vtkstd::vector<vtkImageReader*> ImageReaders;

    const char* fileName = "D:\\Volumes\\4DUltrasound\\3DDCM002.raw";
    std::cout << "Loading " << std::flush;
    int j = 92;
    for (unsigned int i = 0; i < 50; i ++)
    {
        ImageReaders.push_back(vtkImageReader::New());
        ImageReaders[i]->SetDataScalarTypeToUnsignedChar();
        ImageReaders[i]->SetNumberOfScalarComponents(1);
        ImageReaders[i]->SetDataExtent(0, 79, 0, 79, 0, 159);
        ImageReaders[i]->SetFileDimensionality(3);
        ImageReaders[i]->SetDataSpacing(1.0f, 1.0f, 0.74f);
        ImageReaders[i]->SetHeaderSize(i * 80 * 80 * 160 * sizeof(unsigned char));

        ImageReaders[i]->SetFileName(fileName);
        ImageReaders[i]->Update();
        std::cout << "." << std::flush;

        if (ThreadRunning == false)
            return;
    }
    std::cout << "done" <<  std::endl;

    unsigned int frameNumber  = 0;
    while(this->ThreadRunning)
    {
        frameNumber ++;
        if (frameNumber > ImageReaders.size())
            frameNumber = 0;

        this->GetDataInHiddenBuffer()->DeepCopy(ImageReaders[frameNumber]->GetOutput());

        this->SwapBuffers();
    }
}

void vtkUltrasoundScannerReader::StartScanning()
{
    if (this->Thread == NULL)
    {
        this->Thread = vtkMultiThreader::New();
    }
    this->Thread->SetSingleMethod(vtkUltrasoundScannerReader::StartThread, this);
    this->Thread->SingleMethodExecute();

}

void vtkUltrasoundScannerReader::StopScanning()
{
    if (this->Thread != NULL)
    {
        this->ThreadRunning = false;
    }
}

void vtkUltrasoundScannerReader::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}
