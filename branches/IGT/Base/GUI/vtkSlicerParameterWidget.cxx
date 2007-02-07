#include "vtkSlicerParameterWidget.h"

#include <vtkObjectFactory.h>
#include <vtkKWWidget.h>

#include "vtkCommand.h"
#include "vtkSmartPointer.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkSlicerModuleLogic.h"
#include "vtkKWScaleWithEntry.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWMenuButton.h"
#include "vtkKWScale.h"
#include "vtkKWMenu.h"
#include "vtkKWEntry.h"
#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWPushButton.h"
#include "vtkKWLabel.h"
#include "vtkKWLabelWithLabel.h"
#include "vtkKWSpinBox.h"
#include "vtkKWSpinBoxWithLabel.h"
#include "vtkKWCheckButton.h"
#include "vtkKWCheckButtonWithLabel.h"
#include "vtkKWLoadSaveButton.h"
#include "vtkKWLoadSaveButtonWithLabel.h"
#include "vtkKWLoadSaveDialog.h"
#include "vtkKWRadioButton.h"
#include "vtkKWRadioButtonSet.h"
#include "vtkKWRadioButtonSetWithLabel.h"
#include "vtkKWText.h"
#include "vtkKWTextWithScrollbars.h"
#include "vtkKWMessage.h"
#include "vtkKWProgressGauge.h"
#include "vtkKWWindowBase.h"

#include "itkNumericTraits.h"

#include <ModuleDescription.h>
#include <ModuleParameterGroup.h>
#include <ModuleParameter.h>

#include <string>
#include <vector>

//----------------------------------------------------------------------------
vtkStandardNewMacro( vtkSlicerParameterWidget );
//vtkCxxRevisionMacro(vtkSlicerParameteWidget, "$Revision: 1.33 $");
//----------------------------------------------------------------------------

vtkSlicerParameterWidget::vtkSlicerParameterWidget()
{
    this->m_Created = false;
    this->m_CurrentIndex = -1;
    this->m_End = true;
    this->m_InternalWidgetParamList = NULL;
    this->m_ModuleDescription = NULL;
    this->m_ModuleLogic = NULL;
    this->m_ParentWidget = NULL;
}

vtkSlicerParameterWidget::~vtkSlicerParameterWidget()
{
}

void vtkSlicerParameterWidget::CreateWidgets()
{
    if(!this->GetApplication())
    {
        std::cout<<"vtkSlicerParameterWidget: Application is not set!"<<std::endl;
        return;
    }    
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    
    if(!this->m_ModuleDescription)
    {
        std::cout<<"vtkSlicerParameterWidget: ModuleDescription is not set!"<<std::endl;
        return;
    }
    
    if(!this->m_ParentWidget)
    {
        std::cout<<this->GetApplication()->GetNumberOfWindows()<<std::endl;
        vtkKWWindowBase *parentWindows = this->GetApplication()->GetNthWindow(0);
        vtkKWWidget *parentWidget = parentWindows->GetViewFrame();
        if(!parentWidget)
            return;
        else
            this->m_ParentWidget = parentWidget;
    }
    
    std::string title = this->m_ModuleDescription->GetTitle();
    
    vtkSlicerParameterWidget::moduleParameterWidgetStruct *curModWidgetStruct;
    
    this->m_InternalWidgetParamList = new std::vector<vtkSlicerParameterWidget::moduleParameterWidgetStruct*>;
    
    // iterate over each parameter group
    std::vector<ModuleParameterGroup>::const_iterator pgbeginit
      = this->m_ModuleDescription->GetParameterGroups().begin();
    std::vector<ModuleParameterGroup>::const_iterator pgendit
      = this->m_ModuleDescription->GetParameterGroups().end();
    std::vector<ModuleParameterGroup>::const_iterator pgit;
    
    for (pgit = pgbeginit; pgit != pgendit; ++pgit)
    {
        curModWidgetStruct = new vtkSlicerParameterWidget::moduleParameterWidgetStruct; 
        curModWidgetStruct->modParam = NULL;
        curModWidgetStruct->paramWidget = NULL;
        
      // each parameter group is its own labeled frame
        vtkKWFrame *parameterGroupFrame = vtkKWFrame::New ( );
      parameterGroupFrame->SetParent ( this->m_ParentWidget );
      parameterGroupFrame->Create ( );
//      parameterGroupFrame->SetLabelText ((*pgit).GetLabel().c_str());
//      if ((*pgit).GetAdvanced() == "true")
//        {
//        parameterGroupFrame->CollapseFrame ( );
//        }
      
      std::string parameterGroupBalloonHelp = (*pgit).GetDescription();
      parameterGroupFrame
        ->SetBalloonHelpString(parameterGroupBalloonHelp.c_str());

//      app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
//                    parameterGroupFrame->GetWidgetName() );

      // Store the parameter group frame in a SmartPointer
//      (*this->InternalWidgetMap)[(*pgit).GetLabel()] = parameterGroupFrame;
        
      parameterGroupFrame->Delete();
      
      // iterate over each parameter in this group
      std::vector<ModuleParameter>::const_iterator pbeginit
        = (*pgit).GetParameters().begin();
      std::vector<ModuleParameter>::const_iterator pendit
        = (*pgit).GetParameters().end();
      std::vector<ModuleParameter>::const_iterator pit;

      int pcount;
      for (pcount = 0, pit = pbeginit; pit != pendit; ++pit, ++pcount)
      {
        // switch on the type of the parameter...
        vtkKWCoreWidget *parameter;

        if ((*pit).GetTag() == "integer")
          {
          if ((*pit).GetConstraints() == "")
            {
            vtkKWSpinBoxWithLabel *tparameter = vtkKWSpinBoxWithLabel::New();
            tparameter->SetParent( parameterGroupFrame );
            tparameter->Create();
            tparameter->SetLabelText((*pit).GetLabel().c_str());
            tparameter->GetWidget()->SetRestrictValueToInteger();
            tparameter->GetWidget()->SetIncrement(1);
            tparameter->GetWidget()->SetValue(atof((*pit).GetDefault().c_str()));
            tparameter->GetWidget()
              ->SetRange(itk::NumericTraits<int>::NonpositiveMin(),
                         itk::NumericTraits<int>::max());
            parameter = tparameter;
            }
          else
            {
            int min, max, step;
            if ((*pit).GetMinimum() != "")
              {
              min = atoi((*pit).GetMinimum().c_str());
              }
            else
              {
              min = itk::NumericTraits<int>::NonpositiveMin();
              }
            if ((*pit).GetMaximum() != "")
              {
              max = atoi((*pit).GetMaximum().c_str());
              }
            else
              {
              max = itk::NumericTraits<int>::max();
              }
            if ((*pit).GetStep() != "")
              {
              step = atoi((*pit).GetStep().c_str());
              }
            else
              {
              step = 1;
              }

            vtkKWScaleWithEntry *tparameter = vtkKWScaleWithEntry::New();
            tparameter->SetParent( parameterGroupFrame );
            tparameter->PopupModeOn();
            tparameter->Create();
            tparameter->SetLabelText((*pit).GetLabel().c_str());
            tparameter->RangeVisibilityOn();
            tparameter->SetRange(min, max);
            tparameter->SetValue(atof((*pit).GetDefault().c_str()));
            tparameter->SetResolution(step);
            parameter = tparameter;
            }
          }
        else if ((*pit).GetTag() == "boolean")
          {
          vtkKWCheckButtonWithLabel *tparameter = vtkKWCheckButtonWithLabel::New();
          tparameter->SetParent( parameterGroupFrame );
          tparameter->Create();
          tparameter->SetLabelText((*pit).GetLabel().c_str());
          tparameter->GetWidget()->SetSelectedState((*pit).GetDefault() == "true" ? 1 : 0);
          parameter = tparameter;
          }
        else if ((*pit).GetTag() == "float")
          {
          if ((*pit).GetConstraints() == "")
            {
            vtkKWSpinBoxWithLabel *tparameter = vtkKWSpinBoxWithLabel::New();
            tparameter->SetParent( parameterGroupFrame );
            tparameter->Create();
            tparameter->SetLabelText((*pit).GetLabel().c_str());
            tparameter->GetWidget()
              ->SetRange(itk::NumericTraits<float>::NonpositiveMin(),
                         itk::NumericTraits<float>::max());
            tparameter->GetWidget()->SetIncrement( 0.1 );
            tparameter->GetWidget()->SetValueFormat("%1.1f");
            tparameter->GetWidget()->SetValue(atof((*pit).GetDefault().c_str()));
            parameter = tparameter;
            }
          else
            {
            double min, max, step;
            if ((*pit).GetMinimum() != "")
              {
              min = atof((*pit).GetMinimum().c_str());
              }
            else
              {
              min = itk::NumericTraits<float>::NonpositiveMin();
              }
            if ((*pit).GetMaximum() != "")
              {
              max = atof((*pit).GetMaximum().c_str());
              }
            else
              {
              max = itk::NumericTraits<float>::max();
              }
            if ((*pit).GetStep() != "")
              {
              step = atof((*pit).GetStep().c_str());
              }
            else
              {
              step = 0.1;
              }

            vtkKWScaleWithEntry *tparameter
              = vtkKWScaleWithEntry::New();
            tparameter->SetParent( parameterGroupFrame );
            tparameter->PopupModeOn();
            tparameter->Create();
            tparameter->SetLabelText((*pit).GetLabel().c_str());
            tparameter->RangeVisibilityOn();
            tparameter->SetRange(min, max);
            tparameter->SetResolution(step);
            tparameter->SetValue(atof((*pit).GetDefault().c_str()));
            parameter = tparameter;
            }
          }
        else if ((*pit).GetTag() == "double")
          {
          if ((*pit).GetConstraints() == "")
            {
            vtkKWSpinBoxWithLabel *tparameter = vtkKWSpinBoxWithLabel::New();
            tparameter->SetParent( parameterGroupFrame );
            tparameter->Create();
            tparameter->SetLabelText((*pit).GetLabel().c_str());
            tparameter->GetWidget()
              ->SetRange(itk::NumericTraits<double>::NonpositiveMin(),
                         itk::NumericTraits<double>::max());
            tparameter->GetWidget()->SetIncrement( 0.1 );
            tparameter->GetWidget()->SetValueFormat("%1.1f");
            tparameter->GetWidget()->SetValue(atof((*pit).GetDefault().c_str()));
            parameter = tparameter;
            }
          else
            {
            double min, max, step;
            if ((*pit).GetMinimum() != "")
              {
              min = atof((*pit).GetMinimum().c_str());
              }
            else
              {
              min = itk::NumericTraits<double>::NonpositiveMin();
              }
            if ((*pit).GetMaximum() != "")
              {
              max = atof((*pit).GetMaximum().c_str());
              }
            else
              {
              max = itk::NumericTraits<double>::max();
              }
            if ((*pit).GetStep() != "")
              {
              step = atof((*pit).GetStep().c_str());
              }
            else
              {
              step = 0.1;
              }

            vtkKWScaleWithEntry *tparameter
              = vtkKWScaleWithEntry::New();
            tparameter->SetParent( parameterGroupFrame );
            tparameter->PopupModeOn();
            tparameter->Create();
            tparameter->SetLabelText((*pit).GetLabel().c_str());
            tparameter->RangeVisibilityOn();
            tparameter->SetRange(min, max);
            tparameter->SetResolution(step);
            tparameter->SetValue(atof((*pit).GetDefault().c_str()));
            parameter = tparameter;
            }
          }
        else if ((*pit).GetTag() == "string"
                 || (*pit).GetTag() == "integer-vector"
                 || (*pit).GetTag() == "float-vector"
                 || (*pit).GetTag() == "double-vector"
                 || (*pit).GetTag() == "string-vector")
          {
          vtkKWEntryWithLabel *tparameter = vtkKWEntryWithLabel::New();
          tparameter->SetParent( parameterGroupFrame );
          tparameter->Create();
          tparameter->SetLabelText((*pit).GetLabel().c_str());
          tparameter->GetWidget()->SetValue((*pit).GetDefault().c_str());
          parameter = tparameter;
          }
        else if ((*pit).GetTag() == "point")
          {
          vtkSlicerNodeSelectorWidget *tparameter
            = vtkSlicerNodeSelectorWidget::New();
          
          tparameter->SetNodeClass("vtkMRMLFiducialListNode",
                                   NULL,
                                   NULL,
                                   (title + " FiducialList").c_str());
          tparameter->SetNewNodeEnabled(1);
          tparameter->SetNoneEnabled(1);
//          tparameter->SetNewNodeName((title+" output").c_str());
          tparameter->SetParent( parameterGroupFrame );
          tparameter->Create();
          tparameter->SetMRMLScene(this->m_ModuleLogic->GetMRMLScene());
          tparameter->UpdateMenu();
          
          tparameter->SetBorderWidth(2);
          tparameter->SetReliefToFlat();
          tparameter->SetLabelText( (*pit).GetLabel().c_str());
          parameter = tparameter;
          }
        else if ((*pit).GetTag() == "image" && (*pit).GetChannel() == "input")
          {
          vtkSlicerNodeSelectorWidget *tparameter
            = vtkSlicerNodeSelectorWidget::New();
          std::string labelAttrName("LabelMap");
          std::string labelAttrValue("1");
          std::string nodeClass;
          const char *attrName = 0;
          const char *attrValue = 0;
          if ((*pit).GetType() == "label")
            {
            nodeClass = "vtkMRMLScalarVolumeNode";
            attrName = labelAttrName.c_str();
            attrValue = labelAttrValue.c_str();
            }
          else if ((*pit).GetType() == "vector")
            {
            nodeClass = "vtkMRMLVectorVolumeNode";
            }
          else if ((*pit).GetType() == "tensor")
            {
            nodeClass = "vtkMRMLDiffusionTensorVolumeNode";
            }
          else if ((*pit).GetType() == "diffusion-weighted")
            {
            nodeClass = "vtkMRMLDiffusionWeightedVolumeNode";
            }
          else
            {
            nodeClass = "vtkMRMLScalarVolumeNode";
            }

          tparameter->SetNodeClass(nodeClass.c_str(), attrName, attrValue, 
                                   (title + " Volume").c_str());
          tparameter->SetParent( parameterGroupFrame );
          tparameter->Create();
          tparameter->SetMRMLScene(this->m_ModuleLogic->GetMRMLScene());
          tparameter->UpdateMenu();
          
          tparameter->SetBorderWidth(2);
          tparameter->SetReliefToFlat();
          tparameter->SetLabelText( (*pit).GetLabel().c_str());
          parameter = tparameter;
          }
        else if ((*pit).GetTag() == "image" && (*pit).GetChannel() == "output")
          {
          vtkSlicerNodeSelectorWidget *tparameter
            = vtkSlicerNodeSelectorWidget::New();
          std::string labelAttrName("LabelMap");
          std::string labelAttrValue("1");
          std::string nodeClass;
          const char *attrName = 0;
          const char *attrValue = 0;
          if ((*pit).GetType() == "label")
            {
            nodeClass = "vtkMRMLScalarVolumeNode";
            attrName = labelAttrName.c_str();
            attrValue = labelAttrValue.c_str();
            }
          else if ((*pit).GetType() == "vector")
            {
            nodeClass = "vtkMRMLVectorVolumeNode";
            }
          else if ((*pit).GetType() == "tensor")
            {
            nodeClass = "vtkMRMLDiffusionTensorVolumeNode";
            }
          else if ((*pit).GetType() == "diffusion-weighted")
            {
            nodeClass = "vtkMRMLDiffusionWeightedVolumeNode";
            }
          else
            {
            nodeClass = "vtkMRMLScalarVolumeNode";
            }

          tparameter->SetNodeClass(nodeClass.c_str(), attrName, attrValue, 
                                   (title + " Volume").c_str());
          tparameter->SetNewNodeEnabled(1);
          tparameter->SetNoneEnabled(1);
//          tparameter->SetNewNodeName((title+" output").c_str());
          tparameter->SetParent( parameterGroupFrame );
          tparameter->Create();
          tparameter->SetMRMLScene(this->m_ModuleLogic->GetMRMLScene());
          tparameter->UpdateMenu();
          
          tparameter->SetBorderWidth(2);
          tparameter->SetReliefToFlat();
          tparameter->SetLabelText( (*pit).GetLabel().c_str());
          parameter = tparameter;
          }
        else if ((*pit).GetTag() == "geometry" && (*pit).GetChannel() == "input")
          {
          vtkSlicerNodeSelectorWidget *tparameter
            = vtkSlicerNodeSelectorWidget::New();
          tparameter->SetNodeClass("vtkMRMLModelNode",
                                   NULL,
                                   NULL,
                                   (title + " Model").c_str());
          tparameter->SetParent( parameterGroupFrame );
          tparameter->Create();
          tparameter->SetMRMLScene(this->m_ModuleLogic->GetMRMLScene());
          tparameter->UpdateMenu();
          
          tparameter->SetBorderWidth(2);
          tparameter->SetReliefToFlat();
          tparameter->SetLabelText( (*pit).GetLabel().c_str());
          parameter = tparameter;
          }
        else if ((*pit).GetTag() == "geometry" && (*pit).GetChannel() =="output")
          {
          vtkSlicerNodeSelectorWidget *tparameter
            = vtkSlicerNodeSelectorWidget::New();
          
          tparameter->SetNodeClass("vtkMRMLModelNode",
                                   NULL,
                                   NULL,
                                   (title + " Model").c_str());
          tparameter->SetNewNodeEnabled(1);
          tparameter->SetNoneEnabled(1);
//          tparameter->SetNewNodeName((title+" output").c_str());
          tparameter->SetParent( parameterGroupFrame );
          tparameter->Create();
          tparameter->SetMRMLScene(this->m_ModuleLogic->GetMRMLScene());
          tparameter->UpdateMenu();
          
          tparameter->SetBorderWidth(2);
          tparameter->SetReliefToFlat();
          tparameter->SetLabelText( (*pit).GetLabel().c_str());
          parameter = tparameter;
          }
        else if ((*pit).GetTag() == "directory")
          {
          vtkKWLoadSaveButtonWithLabel *tparameter
            = vtkKWLoadSaveButtonWithLabel::New();
          tparameter->SetParent( parameterGroupFrame );
          tparameter->Create();
          if ((*pit).GetChannel() == "output")
            {
            tparameter->GetWidget()->GetLoadSaveDialog()->SaveDialogOn();
            }
          tparameter->SetLabelText( (*pit).GetLabel().c_str() );
          tparameter->GetWidget()->GetLoadSaveDialog()->ChooseDirectoryOn();
          tparameter->GetWidget()->GetLoadSaveDialog()->SetInitialFileName( (*pit).GetDefault().c_str() );
          parameter = tparameter;
          }
        else if ((*pit).GetTag() == "file")
          {
          vtkKWLoadSaveButtonWithLabel *tparameter
            = vtkKWLoadSaveButtonWithLabel::New();
          tparameter->SetParent( parameterGroupFrame );
          if ((*pit).GetChannel() == "output")
            {
            tparameter->GetWidget()->GetLoadSaveDialog()->SaveDialogOn();
            }
          tparameter->Create();
          tparameter->SetLabelText( (*pit).GetLabel().c_str() );
          tparameter->GetWidget()->GetLoadSaveDialog()->SetInitialFileName( (*pit).GetDefault().c_str() );
          parameter = tparameter;
          }
        else if ((*pit).GetTag() == "string-enumeration"
                 || (*pit).GetTag() == "integer-enumeration"
                 || (*pit).GetTag() == "float-enumeration"
                 || (*pit).GetTag() == "double-enumeration")
          {
          vtkKWRadioButtonSetWithLabel *tparameter
            = vtkKWRadioButtonSetWithLabel::New();
          tparameter->SetParent( parameterGroupFrame );
          tparameter->Create();
          tparameter->SetLabelText( (*pit).GetLabel().c_str() );
          tparameter->GetWidget()->PackHorizontallyOn();
          tparameter->GetWidget()->SetMaximumNumberOfWidgetsInPackingDirection(4);
          std::vector<std::string>::const_iterator sbeginit
            = (*pit).GetElements().begin();
          std::vector<std::string>::const_iterator sendit
            = (*pit).GetElements().end();
          std::vector<std::string>::const_iterator sit;
          int id;
          for(sit = sbeginit, id=0; sit != sendit; ++sit, ++id)
            {
            vtkKWRadioButton *b = tparameter->GetWidget()->AddWidget(id);
            b->SetValue( (*sit).c_str() );
            b->SetText( (*sit).c_str() );
            b->SetAnchorToWest();
            if (*sit == (*pit).GetDefault())
              {
              b->SetSelectedState(1);
              }
            else
              {
              b->SetSelectedState(0);
              }
            }
          parameter = tparameter;
          }
        else
          {
          vtkKWLabel *tparameter = vtkKWLabel::New();
          tparameter->SetParent( parameterGroupFrame );
          tparameter->Create();
          tparameter->SetText( (*pit).GetLabel().c_str() );
          parameter = tparameter;
          }

        // build the balloon help for the parameter
        std::string parameterBalloonHelp = (*pit).GetDescription();
        parameter->SetBalloonHelpString(parameterBalloonHelp.c_str());

        // pack the parameter. if the parameter has a separate label and
        // widget, then pack both side by side.
        app->Script ( "pack %s -side top -anchor ne -padx 2 -pady 2",
                      parameter->GetWidgetName() );

        // Store the parameter widget in a SmartPointer
//        (*this->InternalWidgetMap)[(*pit).GetName()] = parameter;
        parameter->Delete();
        }
      curModWidgetStruct->modParam = &(*pit);
      curModWidgetStruct->paramWidget = parameterGroupFrame;
      this->m_InternalWidgetParamList->push_back(curModWidgetStruct);
    }//for
    this->m_Created = true;
    this->m_End = false;
}

vtkKWWidget *vtkSlicerParameterWidget::GetNextWidget()
{
    if(!this->IsCreated() || this->m_CurrentIndex == (this->m_InternalWidgetParamList->size() - 1))
    {
        return NULL;
    }
    
    if(this->m_CurrentIndex == -1)
    {
        this->m_CurrentIndex = 0;
        this->m_InternalIterator = this->m_InternalWidgetParamList->begin();
    }
    else if(this->m_InternalIterator != this->m_InternalWidgetParamList->end())
    {
        this->m_CurrentIndex++;
        this->m_InternalIterator++;    
    }
    else
    {
        this->m_End = true;
        return NULL;
    }        
    return (*(this->m_InternalIterator))->paramWidget;
}

bool vtkSlicerParameterWidget::IsCreated()
{
    return this->m_Created;
}

int vtkSlicerParameterWidget::size()
{
    if(IsCreated())
    {
        return this->m_InternalWidgetParamList->size();   
    }
    else
        return -1;
}

bool vtkSlicerParameterWidget::end()
{
    return this->m_End;
}

void vtkSlicerParameterWidget::reset()
{
    this->m_CurrentIndex = -1;
//    this->m_InternalIterator = NULL;
    this->m_End = false;
}

const ModuleParameter *vtkSlicerParameterWidget::GetCurrentParameter()
{
    if(this->end())
    {
        return NULL;
    }
    else
    {
        return (*(this->m_InternalIterator))->modParam;
    }
}

int vtkSlicerParameterWidget::currentIndex()
{
    return this->m_CurrentIndex;
}

void vtkSlicerParameterWidget::SetParent(vtkKWWidget *parentWidget)
{
    this->m_ParentWidget = parentWidget;
}

void vtkSlicerParameterWidget::SetSlicerModuleLogic(vtkSlicerModuleLogic *moduleLogic)
{
    this->m_ModuleLogic = moduleLogic;
}

void vtkSlicerParameterWidget::SetModuleDescription(ModuleDescription *modDescription)
{
    this->m_ModuleDescription = modDescription;
}
