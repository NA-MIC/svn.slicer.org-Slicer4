
#include "vtkIGTDataManager.h"
#include "vtkObjectFactory.h"
#include "vtkMRMLModelNode.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkMRMLLinearTransformNode.h"
#include "vtkCylinderSource.h"
#include "vtkSphereSource.h"
#include "vtkAppendPolyData.h"




vtkStandardNewMacro(vtkIGTDataManager);
vtkCxxRevisionMacro(vtkIGTDataManager, "$Revision: 1.0 $");

int vtkIGTDataManager::index = 0;

vtkIGTDataManager::vtkIGTDataManager()
{
    this->MRMLScene = NULL;
}


vtkIGTDataManager::~vtkIGTDataManager()
{

}


const char *vtkIGTDataManager::RegisterStream(int streamType)
{

    this->StreamID = "";

    // streamType: 0 - matrix; 1 - image 
    switch (streamType) {
        case IGT_MATRIX_STREAM:
            {

            vtkMRMLModelNode *modelNode = vtkMRMLModelNode::New();
            vtkMRMLModelDisplayNode *dispNode = vtkMRMLModelDisplayNode::New();
            vtkMRMLLinearTransformNode *transform = vtkMRMLLinearTransformNode::New();
            transform->SetHideFromEditors(1);
            transform->SetName("IGTDataManagerTransform");
            dispNode->SetVisibility(0);

            this->MRMLScene->SaveStateForUndo();
            this->MRMLScene->AddNode(dispNode);
            this->MRMLScene->AddNode(transform);
            this->MRMLScene->AddNode(modelNode);  

            dispNode->SetScene(this->MRMLScene);

            char name[20];
            sprintf(name, "igt_matrix_%d", index);

            modelNode->SetName(name);
            modelNode->SetHideFromEditors(1);
            modelNode->SetScene(this->MRMLScene);
            modelNode->SetAndObserveDisplayNodeID(dispNode->GetID());  
            modelNode->SetAndObserveTransformNodeID(transform->GetID());  
            this->StreamID = std::string(modelNode->GetID());

            // Cylinder represents the locator stick
            vtkCylinderSource *cylinder = vtkCylinderSource::New();
            cylinder->SetRadius(1.5);
            cylinder->SetHeight(100);
            cylinder->Update();
            // Sphere represents the locator tip 
            vtkSphereSource *sphere = vtkSphereSource::New();
            sphere->SetRadius(3.0);
            sphere->SetCenter(0, -50, 0);
            sphere->Update();

            vtkAppendPolyData *apd = vtkAppendPolyData::New();
            apd->AddInput(sphere->GetOutput());
            apd->AddInput(cylinder->GetOutput());
            apd->Update();

            modelNode->SetAndObservePolyData(apd->GetOutput());
            this->Modified();  
            this->MRMLScene->Modified();

            modelNode->Delete();
            cylinder->Delete();
            sphere->Delete();
            apd->Delete();
            dispNode->Delete();
            transform->Delete();
            }
            break;

        case IGT_IMAGE_STREAM:
            break;
        default:
            break;
    }

    return this->StreamID.c_str();
}



//Philip Mewes: 07/20/2007 This Locator has an extention to show were the needle is going to advance

const char *vtkIGTDataManager::RegisterStream_new(int streamType)
{

    this->StreamID = "";

    // streamType: 0 - matrix; 1 - image 
    switch (streamType) {
        case IGT_MATRIX_STREAM:
            {

            vtkMRMLModelNode *modelNode_extend = vtkMRMLModelNode::New();
            vtkMRMLModelDisplayNode *dispNode_extend = vtkMRMLModelDisplayNode::New();
            vtkMRMLLinearTransformNode *transform_extend = vtkMRMLLinearTransformNode::New();
            dispNode_extend->SetVisibility(0);
            dispNode_extend->GetColorNode();
           
            dispNode_extend->SetColor(0, 100, 0);
            dispNode_extend->SetOpacity(0.5);

            this->MRMLScene->SaveStateForUndo();
            this->MRMLScene->AddNode(dispNode_extend);
            this->MRMLScene->AddNode(transform_extend);
            this->MRMLScene->AddNode(modelNode_extend);  

            dispNode_extend->SetScene(this->MRMLScene);

            char name[20];
            sprintf(name, "igt_matrix_%d", index);

            modelNode_extend->SetName(name);
            modelNode_extend->SetHideFromEditors(1);
            modelNode_extend->SetScene(this->MRMLScene);
            modelNode_extend->SetAndObserveDisplayNodeID(dispNode_extend->GetID());  
            modelNode_extend->SetAndObserveTransformNodeID(transform_extend->GetID());  
            this->StreamID = std::string(modelNode_extend->GetID());

            // Cylinder represents the locator stick
            vtkCylinderSource *cylinder_extend = vtkCylinderSource::New();
            cylinder_extend->SetRadius(1.5);
            cylinder_extend->SetHeight(100);
            cylinder_extend->Update();  
  
            // Sphere represents the locator tip 
            vtkSphereSource *sphere_extend = vtkSphereSource::New();
            sphere_extend->SetRadius(3.0);
            sphere_extend->SetCenter(0, -50, 0);
            sphere_extend->Update();
            
           
            vtkCylinderSource *VirtualCylinder_extend = vtkCylinderSource::New();
            VirtualCylinder_extend->SetRadius(1.0);
            VirtualCylinder_extend->SetHeight(300);
            VirtualCylinder_extend->SetCenter(0, -100, 0);
            VirtualCylinder_extend->SetResolution(50);
            VirtualCylinder_extend->Update();
            

            vtkAppendPolyData *apd_extend = vtkAppendPolyData::New();
            apd_extend->AddInput(sphere_extend->GetOutput());
            apd_extend->AddInput(cylinder_extend->GetOutput());
            apd_extend->AddInput(VirtualCylinder_extend->GetOutput());
            apd_extend->Update();

            modelNode_extend->SetAndObservePolyData(apd_extend->GetOutput());
            this->Modified();  
            this->MRMLScene->Modified();

            modelNode_extend->Delete();
            cylinder_extend->Delete();
            sphere_extend ->Delete();
            VirtualCylinder_extend->Delete();
            apd_extend->Delete();
            dispNode_extend->Delete();
            transform_extend->Delete();
            }
            break;

        case IGT_IMAGE_STREAM:
            break;
        default:
            break;
    }

    return this->StreamID.c_str();
}






void vtkIGTDataManager::PrintSelf(ostream& os, vtkIndent indent)
{
}

