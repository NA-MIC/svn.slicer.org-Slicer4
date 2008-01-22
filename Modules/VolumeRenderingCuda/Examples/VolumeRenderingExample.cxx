
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"

int main(int argc, char** argv)
{
  
  
  vtkRenderer *renderer = vtkRenderer::New();
  vtkRenderWindow *renWin = vtkRenderWindow::New();
    renWin->AddRenderer(renderer);

  vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
    iren->SetRenderWindow(renWin);

  renderer->AddActor(cubeActor);
      renderer->SetActiveCamera(camera);
      renderer->ResetCamera();
      renderer->SetBackground(1,1,1);
  
  renWin->SetSize(300,300);

  // interact with data
  renWin->Render();
  iren->Start();

  // Clean up

  renderer->Delete();
  renWin->Delete();
  iren->Delete();

  return 0;
}
