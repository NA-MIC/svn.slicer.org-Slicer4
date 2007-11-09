#include "vtkSlicerVRGrayscaleHelper.h"
#include "vtkObjectFactory.h"
#include "vtkKWHistogram.h"
#include "vtkKWHistogramSet.h"
#include "vtkVolumeRenderingModuleGUI.h"
#include "vtkVolume.h"
#include "vtkSlicerVolumeTextureMapper3D.h"
#include "vtkSlicerFixedPointVolumeRayCastMapper.h"
#include "vtkTimerLog.h"
#include "vtkRenderer.h"
#include "vtkPointData.h"
#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkKWProgressGauge.h" 
#include "vtkTexture.h"
#include "vtkImageGradientMagnitude.h"
#include "vtkSlicerApplication.h"
#include "vtkKWEvent.h"
#include "vtkKWMenuButtonWithSpinButtonsWithLabel.h"
#include <math.h>



vtkCxxRevisionMacro(vtkSlicerVRGrayscaleHelper, "$Revision: 1.46 $");
vtkStandardNewMacro(vtkSlicerVRGrayscaleHelper);
vtkSlicerVRGrayscaleHelper::vtkSlicerVRGrayscaleHelper(void)
{
    this->DebugOff();
    this->SetTCLDebug(1);
    this->Histograms=NULL;
    this->SVP_VolumeProperty=NULL;
    this->MB_Quality=NULL;
    this->renViewport=NULL;
    this->renPlane=NULL;
    this->MapperTexture=NULL;
    this->MapperRaycast=NULL;
    this->timer=vtkTimerLog::New();
    this->RenderPlane=0;
    this->currentStage=0;
    this->scheduled=0;
    this->EventHandlerID="";
    this->InitialDropLowRes=0.2;
    this->FactorLastLowRes=0;
    this->LastTimeLowRes=0;
    this->LastTimeHighRes=0;
    this->GoalLowResTime=0.05;
    this->PercentageNoChange=0.4;
    this->TimeToWaitForHigherStage=0.1;
    this->NextRenderHighResolution=0;
    this->IgnoreStepZero=0;
    this->Quality=0;
    this->StageZeroEventHandlerID="";
    this->ButtonDown=0;

}

vtkSlicerVRGrayscaleHelper::~vtkSlicerVRGrayscaleHelper(void)
{
    this->Gui->Script("bind all <Any-ButtonPress> {}",this->GetTclName());
    this->Gui->Script("bind all <Any-ButtonRelease> {}",this->GetTclName());
    //Remove Obersvers
    this->MapperRaycast->RemoveObservers(vtkCommand::VolumeMapperComputeGradientsStartEvent,(vtkCommand *)this->VolumeRenderingCallbackCommand);
    this->MapperRaycast->RemoveObservers(vtkCommand::VolumeMapperComputeGradientsProgressEvent,(vtkCommand *) this->VolumeRenderingCallbackCommand);
    this->MapperRaycast->RemoveObservers(vtkCommand::VolumeMapperComputeGradientsEndEvent,(vtkCommand *) this->VolumeRenderingCallbackCommand);
    this->MapperTexture->RemoveObservers(vtkCommand::VolumeMapperComputeGradientsStartEvent,(vtkCommand *)this->VolumeRenderingCallbackCommand);
    this->MapperTexture->RemoveObservers(vtkCommand::VolumeMapperComputeGradientsProgressEvent,(vtkCommand *) this->VolumeRenderingCallbackCommand);
    this->MapperTexture->RemoveObservers(vtkCommand::VolumeMapperComputeGradientsEndEvent,(vtkCommand *) this->VolumeRenderingCallbackCommand);
    this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->RemoveObservers(vtkCommand::AbortCheckEvent,(vtkCommand*)this->VolumeRenderingCallbackCommand);
    //Only needed if using performance enhancement
    this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->RemoveObservers(vtkCommand::StartEvent,(vtkCommand *)this->VolumeRenderingCallbackCommand);
    this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->RemoveObservers(vtkCommand::EndEvent,(vtkCommand *)this->VolumeRenderingCallbackCommand);
    vtkKWRenderWidget *renderWidget=this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer();
    //Remove Observers


    //Remove Volume


    if(this->SVP_VolumeProperty!=NULL)
    {
        this->Gui->Script("pack forget %s",this->SVP_VolumeProperty->GetWidgetName());
        this->SVP_VolumeProperty->SetHistogramSet(NULL);
        this->SVP_VolumeProperty->SetVolumeProperty(NULL);
        this->SVP_VolumeProperty->SetDataSet(NULL);
        this->SVP_VolumeProperty->SetParent(NULL);
        this->SVP_VolumeProperty->RemoveObservers(vtkKWEvent::VolumePropertyChangingEvent,(vtkCommand*)this->VolumeRenderingCallbackCommand);
        this->SVP_VolumeProperty->Delete();
        this->SVP_VolumeProperty=NULL;
    }
    if(this->Histograms!=NULL)
    {
        this->Histograms->RemoveAllHistograms();
        this->Histograms->Delete();
        this->Histograms=NULL;
    }
    //Don't delete the renViewport(allocated in Outside)

    if(this->MapperTexture!=NULL)
    {
        this->MapperTexture->Delete();
        this->MapperTexture=NULL;
    }
    if(this->MapperRaycast!=NULL)
    {
        this->MapperRaycast->Delete();
        this->MapperRaycast=NULL;
    }

    //Ensure that Render is back to normal
    vtkRenderWindow *renWin=this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow();
    //get the viewport renderer up
    this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->RemoveRenderer(this->renPlane);
    this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->AddRenderer(this->renViewport);
    //this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->SetStatusText("");
    this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(0,0);
    this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(1,0);
    this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(2,0);
    //Change viewport(simulation of "sample distance for "rays"" if using texture mapping)
    this->renViewport->SetViewport(0,0,1,1);
    renWin->SwapBuffersOn();

    if(this->renPlane!=NULL)
    {
        this->renPlane->Delete();
        this->renPlane=NULL;
    }
    if(this->timer!=NULL)
    {
        this->timer->StopTimer();
        this->timer->Delete();
        this->timer=NULL;
    }
    if(this->MB_Quality!=NULL)
    {
        this->MB_Quality->SetParent(NULL);
        this->MB_Quality->Delete();
        this->MB_Quality=NULL;
    }
}
void vtkSlicerVRGrayscaleHelper::Init(vtkVolumeRenderingModuleGUI *gui)
{
    Superclass::Init(gui);
    this->Gui->Script("bind all <Any-ButtonPress> {%s SetButtonDown 1}",this->GetTclName());
    this->Gui->Script("bind all <Any-ButtonRelease> {%s SetButtonDown 0}",this->GetTclName());
    this->SVP_VolumeProperty=vtkSlicerVolumePropertyWidget::New();
    this->SVP_VolumeProperty->SetParent(this->Gui->GetdetailsFrame()->GetFrame());
    this->SVP_VolumeProperty->Create();
    this->SVP_VolumeProperty->SetDataSet(vtkMRMLScalarVolumeNode::SafeDownCast(this->Gui->GetNS_ImageData()->GetSelected())->GetImageData());
    this->Histograms=vtkKWHistogramSet::New();

    //Add Histogram for image data
    this->Histograms->AddHistograms(vtkMRMLScalarVolumeNode::SafeDownCast(this->Gui->GetNS_ImageData()->GetSelected())->GetImageData()->GetPointData()->GetScalars());
    //Build the gradient histogram
    vtkImageGradientMagnitude *grad=vtkImageGradientMagnitude::New();
    grad->SetDimensionality(3);
    grad->SetInput(vtkMRMLScalarVolumeNode::SafeDownCast(this->Gui->GetNS_ImageData()->GetSelected())->GetImageData());
    grad->Update();
    vtkKWHistogram *gradHisto=vtkKWHistogram::New();
    gradHisto->BuildHistogram(grad->GetOutput()->GetPointData()->GetScalars(),0);
    this->Histograms->AddHistogram(gradHisto,"0gradient");

    //Delete      
    this->SVP_VolumeProperty->SetHistogramSet(this->Histograms);
    
    //AddEvent for Interactive apply 
    this->SVP_VolumeProperty->AddObserver(vtkKWEvent::VolumePropertyChangingEvent,(vtkCommand*)this->VolumeRenderingCallbackCommand);
    grad->Delete();
    gradHisto->Delete(); 

    //Quality Button
    this->MB_Quality=vtkKWMenuButtonWithSpinButtonsWithLabel::New();
    this->MB_Quality->SetParent(this->Gui->GetdetailsFrame()->GetFrame());
    this->MB_Quality->Create();
    this->MB_Quality->SetLabelText("Performance / Quality:");
    this->MB_Quality->GetWidget()->GetWidget()->GetMenu()->AddRadioButton("Low Quality - High Performance");
    this->MB_Quality->GetWidget()->GetWidget()->GetMenu()->SetItemCommand(0,this,"SetQuality 0");
    this->MB_Quality->GetWidget()->GetWidget()->GetMenu()->AddRadioButton("Middle Quality - Middle Performance");
    this->MB_Quality->GetWidget()->GetWidget()->GetMenu()->SetItemCommand(1,this,"SetQuality 1");
    this->MB_Quality->GetWidget()->GetWidget()->GetMenu()->AddRadioButton("High Quality - Low Performance");
    this->MB_Quality->GetWidget()->GetWidget()->GetMenu()->SetItemCommand(2,this,"SetQuality 2");
    this->MB_Quality->GetWidget()->GetWidget()->GetMenu()->SelectItem(0);

    this->Gui->GetApplicationGUI()->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",this->MB_Quality->GetWidgetName());
    ((vtkSlicerApplication *)this->Gui->GetApplication())->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",this->SVP_VolumeProperty->GetWidgetName());

}
void vtkSlicerVRGrayscaleHelper::InitializePipelineNewCurrentNode()
{
    vtkKWHistogram *histogram=this->Histograms->GetHistogramWithName("0");
    if(histogram==NULL)
    {
        vtkErrorMacro("Problems with HistogramSet");
        return;
    }
    double totalOccurance=histogram->GetTotalOccurence();
    double tresholdLow=totalOccurance*0.2;
    double tresholdHigh=totalOccurance*0.8;
    double range[2];

    histogram->GetRange(range);
    double tresholdLowIndex=range[0];
    double sumLowIndex=0;
    double tresholdHighIndex=range[0];
    double sumHighIndex=0;
    //calculate distance
    double bin_width = (range[1] == range[0] ? 1 :(double)histogram->GetNumberOfBins() / (range[1] - range[0]));
    while (sumLowIndex<tresholdLow)
    {
        sumLowIndex+=histogram->GetOccurenceAtValue(tresholdLowIndex);
        tresholdLowIndex+=bin_width;
    }
    while(sumHighIndex<tresholdHigh)
    {
        sumHighIndex+=histogram->GetOccurenceAtValue(tresholdHighIndex);
        tresholdHighIndex+=bin_width;

    }
    vtkPiecewiseFunction *opacity=this->Gui->GetcurrentNode()->GetVolumeProperty()->GetScalarOpacity();
    opacity->RemoveAllPoints();
    opacity->AddPoint(range[0],0.);
    opacity->AddPoint(tresholdLowIndex,0.0);
    opacity->AddPoint(tresholdHighIndex,0.2);
    opacity->AddPoint(range[1],0.2);
    vtkColorTransferFunction *colorTransfer=this->Gui->GetcurrentNode()->GetVolumeProperty()->GetRGBTransferFunction();
    colorTransfer->RemoveAllPoints();
    colorTransfer->AddRGBPoint(range[0],.3,.3,1.);
    colorTransfer->AddRGBPoint(tresholdLowIndex,.3,.3,1.);
    colorTransfer->AddRGBPoint(tresholdLowIndex+.5*(tresholdHighIndex-tresholdLowIndex),.3,1.,.3);
    colorTransfer->AddRGBPoint(tresholdHighIndex,1.,.3,.3);
    colorTransfer->AddRGBPoint(range[1],1,.3,.3);
    this->UpdateSVP();
}

void vtkSlicerVRGrayscaleHelper::Rendering(void)
{
    if(this->Volume!=NULL)
    {
        vtkErrorMacro("Rendering already called, use update Rendering instead");
        return;
    }

    this->Volume=vtkVolume::New();
    //TODO Dirty fix as Mapper
    if(this->Gui->GetcurrentNode()->GetMapper()==vtkMRMLVolumeRenderingNode::Texture)
    {
        this->MapperTexture=vtkSlicerVolumeTextureMapper3D::New();
        this->MapperTexture->SetSampleDistance(2);
        this->MapperTexture->SetInput(vtkMRMLScalarVolumeNode::SafeDownCast(this->Gui->GetNS_ImageData()->GetSelected())->GetImageData());
        this->Volume->SetMapper(this->MapperTexture);
        //Also take care about Ray Cast
        this->MapperRaycast=vtkSlicerFixedPointVolumeRayCastMapper::New();
        this->MapperRaycast->SetInput(vtkMRMLScalarVolumeNode::SafeDownCast(this->Gui->GetNS_ImageData()->GetSelected())->GetImageData());
        this->MapperRaycast->SetAutoAdjustSampleDistances(0);
        this->MapperRaycast->SetSampleDistance(0.1);
    }
    this->MapperTexture->AddObserver(vtkCommand::VolumeMapperComputeGradientsStartEvent,(vtkCommand *)this->VolumeRenderingCallbackCommand);
    this->MapperTexture->AddObserver(vtkCommand::VolumeMapperComputeGradientsProgressEvent,(vtkCommand *) this->VolumeRenderingCallbackCommand);
    this->MapperTexture->AddObserver(vtkCommand::VolumeMapperComputeGradientsEndEvent,(vtkCommand *) this->VolumeRenderingCallbackCommand);
    this->MapperTexture->AddObserver(vtkCommand::VolumeMapperRenderProgressEvent,(vtkCommand *) this->VolumeRenderingCallbackCommand);
    this->MapperRaycast->AddObserver(vtkCommand::VolumeMapperComputeGradientsStartEvent,(vtkCommand *)this->VolumeRenderingCallbackCommand);
    this->MapperRaycast->AddObserver(vtkCommand::VolumeMapperComputeGradientsProgressEvent,(vtkCommand *) this->VolumeRenderingCallbackCommand);
    this->MapperRaycast->AddObserver(vtkCommand::VolumeMapperComputeGradientsEndEvent,(vtkCommand *) this->VolumeRenderingCallbackCommand);
    this->MapperRaycast->AddObserver(vtkCommand::ProgressEvent,(vtkCommand *) this->VolumeRenderingCallbackCommand);
    //Only needed if using performance enhancement
    this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->AddObserver(vtkCommand::StartEvent,(vtkCommand *)this->VolumeRenderingCallbackCommand);
    this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->AddObserver(vtkCommand::EndEvent,(vtkCommand *)this->VolumeRenderingCallbackCommand);

    //TODO This is not the right place for this
    this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->AddObserver(vtkCommand::AbortCheckEvent,(vtkCommand*)this->VolumeRenderingCallbackCommand);
    this->Volume->SetProperty(this->Gui->GetcurrentNode()->GetVolumeProperty());
    vtkMatrix4x4 *matrix=vtkMatrix4x4::New();
    vtkMRMLScalarVolumeNode::SafeDownCast(this->Gui->GetNS_ImageData()->GetSelected())->GetIJKToRASMatrix(matrix);
    this->Volume->PokeMatrix(matrix);

    //For Performance
    this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->AddViewProp(this->Volume);
    this->renViewport=this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetNthRenderer(0);
    matrix->Delete();
    //Render
    this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->Render();
}

void vtkSlicerVRGrayscaleHelper::UpdateRendering()
{
    //first check if REndering was already called
    if(this->Volume==NULL)
    {
        this->Rendering();
        return;
    }
    //Update mapper
    this->MapperRaycast->SetInput(vtkMRMLScalarVolumeNode::SafeDownCast(this->Gui->GetNS_ImageData()->GetSelected())->GetImageData());
    this->MapperTexture->SetInput(vtkMRMLScalarVolumeNode::SafeDownCast(this->Gui->GetNS_ImageData()->GetSelected())->GetImageData());
    //Update Property
    this->Volume->SetProperty(this->Gui->GetcurrentNode()->GetVolumeProperty());
    //Update matrix
    vtkMatrix4x4 *matrix=vtkMatrix4x4::New();
    vtkMRMLScalarVolumeNode::SafeDownCast(this->Gui->GetNS_ImageData()->GetSelected())->GetIJKToRASMatrix(matrix);
    this->Volume->PokeMatrix(matrix);

    matrix->Delete();
    this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->Render();
}
void vtkSlicerVRGrayscaleHelper::ShutdownPipeline(void)
{

    if(this->MapperRaycast!=NULL)
    {
        this->MapperRaycast->Delete();
        this->MapperRaycast=NULL;
    }
    if(this->MapperTexture!=NULL)
    {
        this->MapperTexture->Delete();
        this->MapperTexture=NULL;
    }
}

void vtkSlicerVRGrayscaleHelper::ProcessVolumeRenderingEvents(vtkObject *caller,unsigned long eid,void *callData)
{
    vtkSlicerVolumePropertyWidget *callerObjectSVP=vtkSlicerVolumePropertyWidget::SafeDownCast(caller);
    if(callerObjectSVP==this->SVP_VolumeProperty&&eid==vtkKWEvent::VolumePropertyChangingEvent)
    {
        this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->Render();
    }

    //TODO not the right place for this
    vtkRenderWindow *callerRen=vtkRenderWindow::SafeDownCast(caller);
    if(caller==this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()&&eid==vtkCommand::AbortCheckEvent)
    {
        this->CheckAbort();
        return;
    }
    else if(caller==this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()&&eid==vtkCommand::StartEvent)
    {

        //First check if we have to abort the ZeroStageRender
        if(strcmp(this->StageZeroEventHandlerID.c_str(),"")!=0)
        {
            const char* result=this->Gui->Script("after cancel %s",this->StageZeroEventHandlerID.c_str());
             vtkSlicerVRHelperDebug("Result cancel startEvent %s",result);
            this->StageZeroEventHandlerID="";
        }
        //It's the first time we render
        if(this->FactorLastLowRes==0)
        {
            this->FactorLastLowRes=this->InitialDropLowRes;

        }
        vtkSlicerVRHelperDebug("startevent scheduled %d",this->scheduled);
        vtkSlicerVRHelperDebug("startevent currentstage %d",this->currentStage);
        vtkSlicerVRHelperDebug("startevent id %s",this->EventHandlerID.c_str());

        //it is not a scheduled event so we use stage the quality stage and abort every existings scheduling
        if(this->scheduled==0||this->ButtonDown==1)
        {
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->SetStatusText("Using LowestResolution");
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(0,100);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(1,1);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(2,1);


            //Check if we have an Event Scheduled, if this is the case abort it
            if(strcmp(this->EventHandlerID.c_str(),"")!=0)
            {
                const char* result=this->Gui->Script("after cancel %s",this->EventHandlerID.c_str());
                const char* resulta=this->Gui->Script("after info");
                vtkSlicerVRHelperDebug("ResultCancel: %s",result);
                vtkSlicerVRHelperDebug("Result info: %s",resulta);
                this->EventHandlerID="";
            }
            this->currentStage=this->Quality;
           

            //We are already in the Rendering Process
            //this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->Render();
        }//if
        if(this->currentStage==0)
        {

                this->Volume->SetMapper(this->MapperTexture);
            //always got to less sample distance
            this->MapperTexture->SetSampleDistance(2);
            this->currentStage=0;
            vtkSlicerVRHelperDebug("Stage 0 started","");
            //Decide if we REnder plane or not
            if(this->RenderPlane==1)
            {
                return;
            }
            //Time to adjust our Factor
            if(this->LastTimeLowRes<(1-this->PercentageNoChange)*this->GoalLowResTime)
            {
                this->FactorLastLowRes=sqrt((this->FactorLastLowRes*this->FactorLastLowRes)+0.2);

            }
            else if(this->LastTimeLowRes>(1+this->PercentageNoChange)*this->GoalLowResTime)
            {
                double tmp=(this->FactorLastLowRes*this->FactorLastLowRes-0.2);
                if(tmp>0)
                {
                    this->FactorLastLowRes=sqrt(tmp);
                }

            }
            if(this->FactorLastLowRes<this->InitialDropLowRes)
            {
                this->FactorLastLowRes=this->InitialDropLowRes;
            }
            if(this->FactorLastLowRes>1)
            {
                this->FactorLastLowRes=1;
                // Maxbe later
                //this->IgnoreStepZero;
            }
            vtkSlicerVRHelperDebug("NEWFACTORLASTLOWRES %f",this->FactorLastLowRes);
            timer->StartTimer();
            vtkRenderWindow *renWin=this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow();

            //We don't like to see, what we are doing here
            renWin->SwapBuffersOff();
            //get the viewport renderer up
            this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->RemoveRenderer(this->renPlane);
            this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->AddRenderer(this->renViewport);
            //Change viewport(simulation of "sample distance for "rays"" if using texture mapping)
            this->renViewport->SetViewport(0,0,this->FactorLastLowRes,this->FactorLastLowRes);
            return;


        }
        //Stage 1
        if(this->currentStage==1)
        {
            this->Volume->SetMapper(MapperTexture);
            this->MapperTexture->SetSampleDistance(.1);

            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(0,100);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(1,1);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(2,1);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->SetStatusText("Using Middle Resolution");
            vtkSlicerVRHelperDebug("Stage 1 started","");
            //Remove plane Renderer and get viewport Renderer Up
            this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->RemoveRenderer(this->renPlane);
            this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->AddRenderer(this->renViewport);
            //Go to Fullsize
            this->renViewport->SetViewport(0,0,1,1);
            this->scheduled=0;
            return;
        }

        //Stage 2
        if(this->currentStage==2)
        {
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(0,100);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(1,100);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(2,1);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->SetStatusText("Using Highest Resolution");
            vtkSlicerVRHelperDebug("Stage 2 started","");
            this->Volume->SetMapper(this->MapperRaycast);

            this->EventHandlerID="";
            this->scheduled=0;
            return;
        }   
    }
    else if(caller==this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()&&eid==vtkCommand::EndEvent)
    {
        //The Rendering has been aborted
        if(this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->GetAbortRender()==1)
        {
             if(strcmp(this->StageZeroEventHandlerID.c_str(),"")!=0)
        {
            const char* result=this->Gui->Script("after cancel %s",this->StageZeroEventHandlerID.c_str());
             vtkSlicerVRHelperDebug("Result cancel endEvent %s",result);
            this->StageZeroEventHandlerID="";
        }
            this->StageZeroEventHandlerID=this->Gui->Script("after 500 %s ScheduleStageZero", this->GetTclName());
            return;
        }
        vtkSlicerVRHelperDebug("endevent scheduled %d",this->scheduled);
        vtkSlicerVRHelperDebug("endevent currentstage %d",this->currentStage);
        if(this->currentStage==1)
        {
            this->EventHandlerID=this->Gui->Script("after 100 %s ScheduleRender",this->GetTclName());
            vtkSlicerVRHelperDebug("Stage 1 ended","");
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(0,100);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(1,100);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(2,1);
            return;
        }
        if(this->currentStage==2)
        {
            //We reached the highest Resolution, no scheduling
            vtkSlicerVRHelperDebug("Stage 2 ended","");
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(0,100);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(1,100);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(2,100);
            return;
        }
        if(this->RenderPlane==1)
        {
            this->RenderPlane=0;
            this->timer->StopTimer();
            this->LastTimeLowRes=this->timer->GetElapsedTime();
            vtkSlicerVRHelperDebug("Last time low %f",this->LastTimeLowRes);
            //It's time to start for the Scheduled Rendering
            this->EventHandlerID=this->Gui->Script("after 100 %s ScheduleRender",this->GetTclName());
            vtkSlicerVRHelperDebug("Stage 0 ended","");
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(0,100);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(1,1);
            this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(2,1);
            return;
        }


        vtkRenderWindow *renWin=this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow();
        //Get current size of window
        int *size=renWin->GetSize();

        //RGB Data from the smaller viewport
        vtkUnsignedCharArray *image=vtkUnsignedCharArray::New();
        renWin->GetRGBACharPixelData(0,0,this->FactorLastLowRes*size[0],this->FactorLastLowRes*size[1],0,image);

        vtkImageData *imageData=vtkImageData::New();
        imageData->GetPointData()->SetScalars(image);
        imageData->SetDimensions(this->FactorLastLowRes*size[0]+1,this->FactorLastLowRes*size[1]+1,1);
        imageData->SetNumberOfScalarComponents(4);
        imageData->SetScalarTypeToUnsignedChar();
        imageData->SetOrigin(.0,.0,.0);
        imageData->SetSpacing(1.,1.,1.);
        //imageData->Update();
        vtkImageExtractComponents *components=vtkImageExtractComponents::New();
        components->SetInput(imageData);
        components->SetComponents(0,1,2);
        if(this->renPlane==NULL)
        {
            this->renPlane=vtkRenderer::New();
        }

        this->renPlane->SetBackground(this->renViewport->GetBackground());
        this->renPlane->SetActiveCamera(this->renViewport->GetActiveCamera());

        //Get Our Renderer Up
        this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->RemoveRenderer(this->renViewport);
        this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->AddRenderer(this->renPlane);

        this->renPlane->SetDisplayPoint(0,0,0.5);
        this->renPlane->DisplayToWorld();
        double coordinatesA[4];
        this->renPlane->GetWorldPoint(coordinatesA);

        this->renPlane->SetDisplayPoint(size[0],0,0.5);
        this->renPlane->DisplayToWorld();
        double coordinatesB[4];
        this->renPlane->GetWorldPoint(coordinatesB);

        this->renPlane->SetDisplayPoint(size[0],size[1],0.5);
        this->renPlane->DisplayToWorld();
        double coordinatesC[4];
        this->renPlane->GetWorldPoint(coordinatesC);

        this->renPlane->SetDisplayPoint(0,size[1],0.5);
        this->renPlane->DisplayToWorld();
        double coordinatesD[4];
        this->renPlane->GetWorldPoint(coordinatesD);

        //Create the Polydata
        vtkPoints *points=vtkPoints::New();
        points->InsertPoint(0,coordinatesA);
        points->InsertPoint(1,coordinatesB);
        points->InsertPoint(2,coordinatesC);
        points->InsertPoint(3,coordinatesD);

        vtkCellArray *polygon=vtkCellArray::New();
        polygon->InsertNextCell(4);
        polygon->InsertCellPoint(0);
        polygon->InsertCellPoint(1);
        polygon->InsertCellPoint(2);
        polygon->InsertCellPoint(3);
        //Take care about Texture coordinates
        vtkFloatArray *textCoords=vtkFloatArray::New();
        textCoords->SetNumberOfComponents(2);
        textCoords->Allocate(8);
        float tc[2];
        tc[0]=0;
        tc[1]=0;
        textCoords->InsertNextTuple(tc);
        tc[0]=1;
        tc[1]=0;
        textCoords->InsertNextTuple(tc);
        tc[0]=1;
        tc[1]=1;
        textCoords->InsertNextTuple(tc);
        tc[0]=0;
        tc[1]=1;
        textCoords->InsertNextTuple(tc);

        vtkPolyData *polydata=vtkPolyData::New();
        polydata->SetPoints(points);
        polydata->SetPolys(polygon);
        polydata->GetPointData()->SetTCoords(textCoords);

        vtkPolyDataMapper *polyMapper=vtkPolyDataMapper::New();
        polyMapper->SetInput(polydata);

        vtkActor *actor=vtkActor::New(); 
        actor->SetMapper(polyMapper);

        //Take care about the texture
        vtkTexture *atext=vtkTexture::New();
        atext->SetInput(components->GetOutput());
        atext->SetInterpolate(1);
        actor->SetTexture(atext);

        //Remove all old Actors
        this->renPlane->RemoveAllViewProps();

        this->renPlane->AddActor(actor);
        //Remove the old Renderer

        renWin->SwapBuffersOn();
        //Do we do that
        this->RenderPlane=1;



        //Delete everything we have done
        image->Delete();
        imageData->Delete();
        components->Delete();
        points->Delete();
        polygon->Delete();
        textCoords->Delete();
        polydata->Delete();
        polyMapper->Delete();
        actor->Delete();
        atext->Delete();
        this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->Render();

    }
    //Check if caller equals mapper
    vtkAbstractMapper *callerMapper=vtkAbstractMapper::SafeDownCast(caller);
    if((this->Volume!=0)&&(callerMapper!=this->Volume->GetMapper()))
    {
        return;
    }
    if(eid==vtkCommand::VolumeMapperComputeGradientsStartEvent)
    {
        this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->SetStatusText("Calculate Gradients");
    }
    else if(eid==vtkCommand::VolumeMapperComputeGradientsEndEvent)
    {
        this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->SetStatusText("Calculate Gradients finished");
    }
    else if(eid==vtkCommand::VolumeMapperComputeGradientsProgressEvent)
    {
        float *progress=(float*)callData;
        this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetValue(100**progress);
    }
    else if (eid==vtkCommand::VolumeMapperRenderProgressEvent&&this->currentStage==1)
    {
        double *progress=(double*)callData;
        this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(1,100**progress);
    }
    else if (eid==vtkCommand::ProgressEvent)
    {
        float *progress=(float*)callData;
        if(*progress==0)
        {
            return;
        }
        this->Gui->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->SetNthValue(2,100**progress);
    }
}

void vtkSlicerVRGrayscaleHelper::ScheduleRender(void)
{
    this->scheduled=1;
    this->currentStage++;
    if(this->currentStage>2||vtkKWTkUtilities::CheckForPendingInteractionEvents(this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow())!=0)
    {
        this->currentStage=0;
        this->scheduled=0;
        //return;
    }
    this->EventHandlerID=this->Gui->Script("after idle [[[$::slicer3::ApplicationGUI GetViewerWidget] GetMainViewer] GetRenderWindow] Render");
    //this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->Render();
}

void vtkSlicerVRGrayscaleHelper::UpdateSVP(void)
{
    //First check if we have a SVP
    if(this->SVP_VolumeProperty==NULL)
    {
        vtkErrorMacro("SVP does not exist");
    }
    //First of all set New Property, Otherwise all Histograms will be overwritten
    //First check if we really need to update
    if(this->SVP_VolumeProperty->GetVolumeProperty()==this->Gui->GetcurrentNode()->GetVolumeProperty())
    {
        this->AdjustMapping();
        this->SVP_VolumeProperty->Update();
        return;
    }
    this->SVP_VolumeProperty->SetVolumeProperty(this->Gui->GetcurrentNode()->GetVolumeProperty());
    this->SVP_VolumeProperty->SetHSVColorSelectorVisibility(1);
    this->SVP_VolumeProperty->Update();
}

void vtkSlicerVRGrayscaleHelper::UpdateGUIElements(void)
{
    Superclass::UpdateGUIElements();
    this->UpdateSVP();
}
void vtkSlicerVRGrayscaleHelper::CheckAbort(void)
{
    int pending=this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->GetEventPending();
    if(pending!=0)
    {
        vtkSlicerVRHelperDebug("got an abort","");
        this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->SetAbortRender(1);
        this->scheduled=0;
        return;
    }
    int pendingGUI=vtkKWTkUtilities::CheckForPendingInteractionEvents(this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow());
    if(pendingGUI!=0)
    {
        vtkSlicerVRHelperDebug("got an abort from gui","");
        this->Gui->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->SetAbortRender(1);
        this->scheduled=0;
        return;

    }
}
void vtkSlicerVRGrayscaleHelper::AdjustMapping(){

    //Update Color    
    vtkColorTransferFunction *functionColor=this->Gui->GetcurrentNode()->GetVolumeProperty()->GetRGBTransferFunction();
    double rangeNew[2];
    vtkMRMLScalarVolumeNode::SafeDownCast(this->Gui->GetNS_ImageData()->GetSelected())->GetImageData()->GetPointData()->GetScalars()->GetRange(rangeNew);
    functionColor->AdjustRange(rangeNew);

    //Update Opacity
    vtkPiecewiseFunction *function=this->Gui->GetcurrentNode()->GetVolumeProperty()->GetScalarOpacity();
    function->AdjustRange(rangeNew);
    //Update
    rangeNew[0]/=4;
    rangeNew[1]/=4;
    function=this->Gui->GetcurrentNode()->GetVolumeProperty()->GetGradientOpacity();
    function->RemovePoint(255);//Remove the standard value
    //this->Histograms->GetHistogramWithName("0gradient")->GetRange(rangeNew);
    function->AdjustRange(rangeNew);
}

void vtkSlicerVRGrayscaleHelper::ScheduleStageZero()
{
     this->StageZeroEventHandlerID=this->Gui->Script("after idle [[[$::slicer3::ApplicationGUI GetViewerWidget] GetMainViewer] GetRenderWindow] Render");
    
}
