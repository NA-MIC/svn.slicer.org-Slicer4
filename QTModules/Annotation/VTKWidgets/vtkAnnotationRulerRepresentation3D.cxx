// AnnotationModule/VTKWidgets includes
#include "VTKWidgets/vtkAnnotationRulerRepresentation3D.h"

// VTK includes
#include <vtkAxisActor2D.h>
#include <vtkDoubleArray.h>
#include <vtkFollower.h>
#include <vtkGlyph3D.h>
#include <vtkHandleRepresentation.h>
#include <vtkMath.h>
#include <vtkObjectFactory.h>
#include <vtkPoints.h>
#include <vtkRenderer.h>
#include <vtkVectorText.h>
#include <vtkWindow.h>

#include <vtkBox.h>
#include <vtkPointData.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTransform.h>
#include <vtkSmartPointer.h>

// Convenient macro
#define VTK_CREATE(type, name) \
  vtkSmartPointer<type> name = vtkSmartPointer<type>::New()

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkAnnotationRulerRepresentation3D);
vtkCxxRevisionMacro (vtkAnnotationRulerRepresentation3D, "$Revision: 1.0 $");

//---------------------------------------------------------------------------
void vtkAnnotationRulerRepresentation3D::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Distance: " << this->m_Distance << endl;
  os << indent << "LabelScaleSpecified: " << (this->LabelScaleSpecified ? "true" : "false") << endl;
  os << indent << "LabelPosition: " << this->LabelPosition << endl;
  os << indent << "MaxTicks: " << this->MaxTicks << endl;
  os << indent << "GlyphScaleSpecified: " << (this->GlyphScaleSpecified ? "true" : "false") << endl;
  os << indent << "GlyphScale: " << this->GlyphScale << endl;
}

//---------------------------------------------------------------------------
vtkAnnotationRulerRepresentation3D::vtkAnnotationRulerRepresentation3D()
{
  this->m_Distance = 0;
  this->GlyphScaleSpecified = false;
  this->LabelPosition = 0.5;
  this->MaxTicks = 99;
  this->GlyphScale = 1.0;
}

//---------------------------------------------------------------------------
vtkAnnotationRulerRepresentation3D::~vtkAnnotationRulerRepresentation3D()
{

}

//----------------------------------------------------------------------
void vtkAnnotationRulerRepresentation3D::BuildRepresentation()
{
  if ( this->GetMTime() > this->BuildTime ||
       this->Point1Representation->GetMTime() > this->BuildTime ||
       this->Point2Representation->GetMTime() > this->BuildTime ||
       this->LabelActor->GetMTime() > this->BuildTime ||
       this->BoundingBox->GetMTime() > this->BuildTime ||
       this->GlyphActor->GetMTime() > this->BuildTime ||
       this->LineActor->GetMTime() > this->BuildTime ||
       (this->Renderer && this->Renderer->GetVTKWindow() &&
        this->Renderer->GetVTKWindow()->GetMTime() > this->BuildTime) )
    {
    this->Superclass::BuildRepresentation();

    // Okay, compute the distance and set the label
    double p1[3], p2[3];

    this->Point1Representation->GetWorldPosition(p1);
    this->Point2Representation->GetWorldPosition(p2);

    if (this->m_Distance > 0)
      {
      this->Distance = this->m_Distance;
      }
    else
      {
      this->Distance = sqrt(vtkMath::Distance2BetweenPoints(p1,p2));
      }


    // Line
    this->LinePoints->SetPoint(0,p1);
    this->LinePoints->SetPoint(1,p2);
    this->LinePoints->Modified();

    // Label
    char string[512];
    sprintf(string, this->LabelFormat, this->Distance);
    this->LabelText->SetText(string);
    this->UpdateLabelActorPosition();
    if (this->Renderer) //make the label face the camera
      {
      this->LabelActor->SetCamera( this->Renderer->GetActiveCamera() );
      }

    if (!this->LabelScaleSpecified)
      {
      // If a font size hasn't been specified by the user, scale the text
      // (font size) according to the length of the line widget.
      this->LabelActor->SetScale(
          this->Distance/20.0, this->Distance/20.0, this->Distance/20.0 );
      }

    // Ticks - generate points that are glyphed
    int i, numTicks;
    double v21[3], x[3];
    v21[0] =  p2[0] - p1[0]; v21[1] =  p2[1] - p1[1]; v21[2] =  p2[2] - p1[2];
    vtkMath::Normalize(v21);
    this->GlyphPoints->Reset();
    this->GlyphPoints->Modified();
    this->GlyphVectors->Reset();
    if (this->GlyphScaleSpecified)
      {
      this->Glyph3D->SetScaleFactor(this->GlyphScale);
      }
    else
      {
      this->Glyph3D->SetScaleFactor(this->Distance/40);
      }

    double distance;
    if ( this->RulerMode ) // specified tick separation
      {
      numTicks = (this->RulerDistance <= 0.0 ? 1 : (this->Distance / this->RulerDistance));
      numTicks = (numTicks > this->MaxTicks ? this->MaxTicks : numTicks);
      distance = this->RulerDistance;
      }
    else //evenly spaced
      {
      numTicks = this->NumberOfRulerTicks;
      distance = this->Distance / (numTicks + 1);
      }
    for (i=1; i <= numTicks; ++i)
      {
      x[0] = p1[0] + i*v21[0]*distance;
      x[1] = p1[1] + i*v21[1]*distance;
      x[2] = p1[2] + i*v21[2]*distance;
      this->GlyphPoints->InsertNextPoint(x);
      this->GlyphVectors->InsertNextTuple(v21);
      }

    this->BuildTime.Modified();
    }
}

//----------------------------------------------------------------------
void vtkAnnotationRulerRepresentation3D::SetDistance(double distance)
{
  this->m_Distance = distance;
}

//----------------------------------------------------------------------
void vtkAnnotationRulerRepresentation3D::SetGlyphScale( double scale )
{
  this->GlyphScale = scale;
  this->GlyphScaleSpecified = true;
}

//----------------------------------------------------------------------------
vtkProperty * vtkAnnotationRulerRepresentation3D::GetLineProperty()
{
  return this->LineActor->GetProperty();
}

//----------------------------------------------------------------------
void vtkAnnotationRulerRepresentation3D::SetLabelPosition(double labelPosition)
{
  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting LabelPosition to " << labelPosition); 

  if (this->LabelPosition == labelPosition)
    {
//    std::cout << "SetLabelPosition: no change: " << this->LabelPosition << std::endl;
    }
  else
    {
    this->LabelPosition = labelPosition;
    }
  this->UpdateLabelActorPosition();
  // this->Modified();
}

//----------------------------------------------------------------------
void vtkAnnotationRulerRepresentation3D::UpdateLabelActorPosition()
{
  if (!this->Point1Representation ||
      !this->Point2Representation)
    {
    return;
    }

  // get the end points
  double p1[3], p2[3];
  this->Point1Representation->GetWorldPosition(p1);
  this->Point2Representation->GetWorldPosition(p2);
  double pos[3];

//  std::cout << "UpdateLabelActorPosition: p1 = " << p1[0] << ", " << p1[1] << ", " << p1[2] << ". p2 = " << p2[0] << ", " << p2[1] << ", " << p2[2] << std::endl;
  // calculate the label position
  /*
  pos[0] = (p1[0]+p2[0])*this->LabelPosition;
  pos[1] = (p1[1]+p2[1])*this->LabelPosition;
  pos[2] = (p1[2]+p2[2])*this->LabelPosition;
  */
  pos[0] = p1[0] + (p2[0] - p1[0]) * this->LabelPosition;
  pos[1] = p1[1] + (p2[1] - p1[1]) * this->LabelPosition;
  pos[2] = p1[2] + (p2[2] - p1[2]) * this->LabelPosition;
  

  // and set it on the actor
  double * actorPos = this->LabelActor->GetPosition();
  //std::cout << "UpdateLabelActorPosition: LabelPos = " << this->LabelPosition << ", current label actor position = "  << actorPos[0] << ", " << actorPos[1] << ", " << actorPos[2] << ", calculated pos = " << pos[0] << ", " << pos[1] << ", " << pos[2] << std::endl;

  double diff = sqrt(vtkMath::Distance2BetweenPoints(pos, actorPos));
  if (diff > 0.001)
    {
    this->LabelActor->SetPosition(pos);
   
//  this->Modified();
    }
  else
    {
    //  std::cout << "\tlabel actor current position not different enough from calculated, diff = " << diff << ", not setting position" << std::endl;
    }
}

//----------------------------------------------------------------------
void vtkAnnotationRulerRepresentation3D::UpdateGlyphPolyData(vtkPolyData *polyData)
{
  if (!polyData)
    {
    vtkErrorMacro("UpdateGlyphPolyData: invalid poly data!");
    return;
    }
  vtkWarningMacro("UpdateGlyphPolyData: not implemented yet!");
  return;
  /*
  this->GlyphPolyData->DeepCopy(polyData);
  this->GlyphPolyData->SetPoints(this->GlyphPoints);
  this->GlyphPolyData->GetPointData()->SetVectors(this->GlyphVectors);

  this->GlyphXForm->SetInput(this->GlyphPolyData);
  vtkSmartPointer<vtkTransform> xform = vtkSmartPointer<vtkTransform>::New();
  this->GlyphXForm->SetTransform(xform);
  
  this->Glyph3D->SetInput(this->GlyphPolyData);
  // set the transform to identity
  this->Glyph3D->SetSourceConnection(this->GlyphXForm->GetOutputPort());  
  */
}
