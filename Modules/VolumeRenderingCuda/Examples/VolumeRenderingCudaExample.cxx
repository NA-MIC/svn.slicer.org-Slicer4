
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkCamera.h"

#include "vtkVolumeCudaMapper.h"
#include "vtkImageReader.h"
#include <sstream>
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"

#include "vtkActor.h"
#include "vtkCamera.h"
#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"



int main(int argc, char** argv)
{

    vtkVolume* volume = vtkVolume::New();
    vtkVolumeCudaMapper* volumeMapper = vtkVolumeCudaMapper::New();

    // Reading in the Data using a ImageReader
    vtkImageReader* reader[5];
    for (unsigned int i = 0; i < 5; i++ ) 
    {
        reader[i]= vtkImageReader::New();
        reader[i]->SetDataScalarTypeToUnsignedChar();
        reader[i]->SetNumberOfScalarComponents(1);
        reader[i]->SetDataExtent(0, 255,
            0, 255, 
            0, 255);
        reader[i]->SetFileDimensionality(3);

        std::stringstream s;
        s << "C:\\heart256-" << i+1 << ".raw";

        reader[i]->SetFileName(s.str().c_str());
        reader[i]->Update();

        volumeMapper->MultiInput[i] = reader[i]->GetOutput();
    }
    volumeMapper->SetInput(reader[0]->GetOutput());

    volume->SetMapper(volumeMapper);
    vtkVolumeProperty* prop = vtkVolumeProperty::New();
    volume->SetProperty(prop);
    prop->Delete();




    vtkRenderer *renderer = vtkRenderer::New();
    vtkRenderWindow *renWin = vtkRenderWindow::New();
    renWin->AddRenderer(renderer);

    renderer->AddVolume(volume);
    vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
    iren->SetRenderWindow(renWin);

    vtkCamera *camera = vtkCamera::New();
    camera->SetPosition(1,1,1);
    camera->SetFocalPoint(0,0,0);

    renderer->SetActiveCamera(camera);
    renderer->ResetCamera();
    renderer->SetBackground(1,1,1);

    renWin->SetSize(300,300);

    // interact with data
    renWin->Render();
    iren->Start();

    // Clean up
    volume->Delete();
    volumeMapper->Delete();

    renderer->Delete();
    renWin->Delete();
    iren->Delete();

    for (unsigned int i = 0; i < 5; i++)
        reader[i]->Delete();
    return 0;
}
