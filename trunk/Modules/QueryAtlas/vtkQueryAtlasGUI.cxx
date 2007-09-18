#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"

#include "vtkKWWidget.h"
#include "vtkKWPushButton.h"
#include "vtkKWCheckButton.h"
#include "vtkKWRadioButton.h"
#include "vtkKWMenu.h"
#include "vtkKWLabel.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWFrame.h"
#include "vtkKWMultiColumnList.h"
#include "vtkKWMultiColumnListWithScrollbars.h"
#include "vtkKWEntry.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWListBox.h"
#include "vtkKWListBoxWithScrollbars.h"
#include "vtkKWLoadSaveButton.h"
#include "vtkKWLoadSaveButtonWithLabel.h"

#include "vtkSlicerModelsGUI.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleLogic.h"
#include "vtkSlicerVisibilityIcons.h"
#include "vtkSlicerToolbarIcons.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkSlicerSliceControllerWidget.h"
#include "vtkSlicerSliceGUI.h"
#include "vtkSlicerToolbarGUI.h"
#include "vtkQueryAtlasGUI.h"
#include "vtkQueryAtlasUseSearchTermWidget.h"
#include "vtkQueryAtlasSearchTermWidget.h"
#include "vtkSlicerPopUpHelpWidget.h"

// for path manipulation
#include "itksys/SystemTools.hxx"


#define QUERIES_FRAME
#define SEARCHTERM_FRAME
#define RESULTS_FRAME
#define LOAD_FRAME
#define ANNO_FRAME
#define ONTOLOGY_FRAME

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkQueryAtlasGUI );
vtkCxxRevisionMacro ( vtkQueryAtlasGUI, "$Revision: 1.0 $");


#define _br 0.85
#define _bg 0.85
#define _bb 0.95

#define _fr 0.5
#define _fg 0.5
#define _fb 0.5


//---------------------------------------------------------------------------
vtkQueryAtlasGUI::vtkQueryAtlasGUI ( )
{
    this->Logic = NULL;

    this->CollaboratorIcons = NULL;
    this->QueryAtlasIcons = NULL;
    this->AnnotationVisibility = 1;
    this->ModelVisibility = 1;
    this->ProcessingMRMLEvent = 0;
    this->SceneClosing = false;
    
#ifdef SEARCHTERM_FRAME
    //---
    // master category switch
    //---
    this->OtherButton = NULL;
    this->StructureButton = NULL;
    this->PopulationButton = NULL;
    this->SpeciesButton = NULL;
    this->SwitchQueryFrame = NULL;

    //---
    // search term other frame
    //---    
    this->OtherFrame = NULL;
    this->OtherListWidget = NULL;

    //---
    // search term species frame
    //---    
    this->SpeciesFrame = NULL;
    this->SpeciesLabel = NULL;
    this->SpeciesNoneButton = NULL;
    this->SpeciesHumanButton = NULL;
    this->SpeciesMouseButton = NULL;
    this->SpeciesMacaqueButton = NULL;

    //---
    // search term population frame
    //---
    this->PopulationFrame = NULL;
    this->DiagnosisMenuButton = NULL;
    this->GenderMenuButton = NULL;
    this->HandednessMenuButton = NULL;
    this->AgeMenuButton = NULL;
    this->AddDiagnosisEntry = NULL;

    //---
    // search term structure frame
    //---
    this->StructureFrame = NULL;
    this->StructureMenuButton = NULL;
    this->StructureListWidget = NULL;
#endif

    //---
    // annotation frame
    //---    
#ifdef ANNO_FRAME
    this->AnnotationVisibilityButton = NULL;
    this->AnnotationTermSetMenuButton = NULL;
    this->ModelVisibilityButton = NULL;
#endif
    
    //---
    // query frame
    //---    
#ifdef QUERIES_FRAME
    this->SearchButton = NULL;
    this->DatabasesMenuButton = NULL;
    this->ResultsWithAnyButton = NULL;
    this->ResultsWithAllButton = NULL;
    this->ResultsWithExactButton = NULL;    
#endif

    //---
    // results frame
    //---
#ifdef RESULTS_FRAME
    this->CurrentResultsList = NULL;
    this->AccumulatedResultsList = NULL;
    this->DeleteAllCurrentResultsButton = NULL;
    this->DeleteCurrentResultButton = NULL;
    this->DeselectAllCurrentResultsButton = NULL;
    this->DeselectAllAccumulatedResultsButton = NULL;
    this->DeleteAllAccumulatedResultsButton = NULL;
    this->DeleteAccumulatedResultButton = NULL;
    this->SaveCurrentResultsButton = NULL;
    this->SaveCurrentSelectedResultsButton = NULL;
    this->SaveAccumulatedResultsButton = NULL;
    this->LoadURIsButton = NULL;
#endif
    this->NumberOfColumns = 2;

    //---
    // ontology frame
    //---
#ifdef ONTOLOGY_FRAME
    this->LocalSearchTermEntry = NULL;
    this->SynonymsMenuButton = NULL;
    this->BIRNLexEntry = NULL;
    this->BIRNLexIDEntry = NULL;
    this->NeuroNamesEntry = NULL;
    this->NeuroNamesIDEntry = NULL;
    this->UMLSCIDEntry = NULL;
    this->AddLocalTermButton = NULL;
    this->AddSynonymButton = NULL;
    this->AddBIRNLexStringButton = NULL;
    this->AddBIRNLexIDButton = NULL;
    this->AddNeuroNamesStringButton = NULL;
    this->AddNeuroNamesIDButton = NULL;
    this->AddUMLSCIDButton = NULL;
    this->BIRNLexHierarchyButton = NULL;
    this->NeuroNamesHierarchyButton = NULL;
    this->UMLSHierarchyButton = NULL;
    this->SavedTerms = NULL;
#endif

    //---
    // load frame
    //---    
#ifdef LOAD_FRAME
    this->FIPSFSButton = NULL;
    this->FIPSFSFrame = NULL;
    this->QdecButton = NULL;
    this->QdecFrame = NULL;
    this->FSasegSelector = NULL;
    this->FSmodelSelector = NULL;
    this->FSbrainSelector = NULL;
    this->FSstatsSelector = NULL;
    this->FSgoButton = NULL;
    this->QdecGetResultsButton = NULL;
    this->QdecScalarSelector = NULL;
    this->QdecModelSelector = NULL;
    this->QdecGoButton = NULL;
#endif
}


//---------------------------------------------------------------------------
vtkQueryAtlasGUI::~vtkQueryAtlasGUI ( )
{

  vtkDebugMacro("vtkQueryAtlasGUI: Tearing down Tcl callbacks \n");
  this->Script ( "QueryAtlasTearDown" );

  this->RemoveMRMLObservers ( );
    this->SetModuleLogic ( NULL );
    //---
    // help and acknowledgment frame
    //---
    if ( this->CollaboratorIcons )
      {
      this->CollaboratorIcons->Delete();
      this->CollaboratorIcons = NULL;
      }
    if ( this->QueryAtlasIcons )
      {
      this->QueryAtlasIcons->Delete();
      this->QueryAtlasIcons = NULL;
      }


    //---
    // load frame
    //---
#ifdef LOAD_FRAME
    if ( this->FIPSFSButton )
      {
      this->FIPSFSButton->SetParent ( NULL );
      this->FIPSFSButton->Delete();
      this->FIPSFSButton = NULL;
      }
    if ( this->FIPSFSFrame )
      {
      this->FIPSFSFrame->SetParent ( NULL );
      this->FIPSFSFrame->Delete();
      this->FIPSFSFrame = NULL;      
      }
    if ( this->QdecButton )
      {
      this->QdecButton->SetParent ( NULL );
      this->QdecButton->Delete();      
      this->QdecButton = NULL;      
      }
    if ( this->QdecFrame )
      {
      this->QdecFrame->SetParent ( NULL );
      this->QdecFrame->Delete();
      this->QdecFrame = NULL;      
      }
    if ( this->FSgoButton )
      {
      this->FSgoButton->SetParent ( NULL );
      this->FSgoButton->Delete();
      this->FSgoButton = NULL;
      }
    if ( this->QdecGoButton )
      {
      this->QdecGoButton->SetParent ( NULL );
      this->QdecGoButton->Delete();
      this->QdecGoButton = NULL;
      }
    if ( this->FSstatsSelector )
      {
      this->FSstatsSelector->SetParent ( NULL );
      this->FSstatsSelector->Delete();
      this->FSstatsSelector = NULL;
      }
    if ( this->FSbrainSelector )
      {
      this->FSbrainSelector->SetParent ( NULL );
      this->FSbrainSelector->Delete();
      this->FSbrainSelector = NULL;
      }
    if ( this->FSasegSelector )
      {
      this->FSasegSelector->SetParent ( NULL );
      this->FSasegSelector->Delete();
      this->FSasegSelector = NULL;
      }
    if ( this->FSmodelSelector )
      {
      this->FSmodelSelector->SetParent ( NULL );
      this->FSmodelSelector->Delete();
      this->FSmodelSelector = NULL;
      }
    if ( this->QdecGetResultsButton )
      {
      this->QdecGetResultsButton->SetParent ( NULL );
      this->QdecGetResultsButton->Delete();
      this->QdecGetResultsButton = NULL;
      }
    if ( this->QdecModelSelector )
      {
      this->QdecModelSelector->SetParent ( NULL );
      this->QdecModelSelector->Delete();
      this->QdecModelSelector = NULL;
      }
    if ( this->QdecScalarSelector )
      {
      this->QdecScalarSelector->SetParent ( NULL );
      this->QdecScalarSelector->Delete();
      this->QdecScalarSelector = NULL;
      }
#endif


    //---
    // annotation frame
    //---
#ifdef ANNO_FRAME
    if ( this->ModelVisibilityButton )
      {
      this->ModelVisibilityButton->SetParent ( NULL );
      this->ModelVisibilityButton->Delete();
      this->ModelVisibilityButton = NULL;
      }
    if ( this->AnnotationVisibilityButton )
      {
      this->AnnotationVisibilityButton->SetParent ( NULL );
      this->AnnotationVisibilityButton->Delete();
      this->AnnotationVisibilityButton = NULL;
      }
    if ( this->AnnotationTermSetMenuButton )
      {
      this->AnnotationTermSetMenuButton->SetParent ( NULL );
      this->AnnotationTermSetMenuButton->Delete();
      this->AnnotationTermSetMenuButton = NULL;      
      }
#endif
    
    //---
    // search term frame
    //---
#ifdef SEARCHTERM_FRAME
    if ( this->SwitchQueryFrame)
      {
      this->SwitchQueryFrame->SetParent ( NULL );
      this->SwitchQueryFrame->Delete();
      this->SwitchQueryFrame = NULL;
      }
    if ( this->OtherButton )
      {
      this->OtherButton->SetParent ( NULL );
      this->OtherButton->Delete();
      this->OtherButton = NULL;   
      }
    if ( this->StructureButton )
      {
      this->StructureButton->SetParent ( NULL );
      this->StructureButton->Delete();
      this->StructureButton = NULL;
      }
    if ( this->PopulationButton )
      {
      this->PopulationButton->SetParent ( NULL );
      this->PopulationButton->Delete();
      this->PopulationButton = NULL;      
      }
    //---
    // search term population panel
    //---
    if ( this->PopulationFrame )
      {
      this->PopulationFrame->SetParent ( NULL );
      this->PopulationFrame->Delete();
      this->PopulationFrame = NULL;
      }
    if ( this->DiagnosisMenuButton )
      {
      this->DiagnosisMenuButton->SetParent ( NULL );
      this->DiagnosisMenuButton->Delete();
      this->DiagnosisMenuButton = NULL;
      }
    if ( this->GenderMenuButton )
      {
      this->GenderMenuButton->SetParent ( NULL );
      this->GenderMenuButton->Delete();
      this->GenderMenuButton = NULL;
      }
    if ( this->HandednessMenuButton )
      {
      this->HandednessMenuButton->SetParent ( NULL );
      this->HandednessMenuButton->Delete();
      this->HandednessMenuButton = NULL;
      }
    if ( this->AgeMenuButton )
      {
      this->AgeMenuButton->SetParent ( NULL );
      this->AgeMenuButton->Delete();
      this->AgeMenuButton  = NULL;
      }
    if ( this->AddDiagnosisEntry )
      {
      this->AddDiagnosisEntry->SetParent ( NULL );
      this->AddDiagnosisEntry->Delete();
      this->AddDiagnosisEntry = NULL;
      }
    //---
    // search term species panel
    //---
    if ( this->SpeciesFrame)
      {
      this->SpeciesFrame->SetParent ( NULL );
      this->SpeciesFrame->Delete();
      this->SpeciesFrame = NULL;
      }
    if ( this->SpeciesButton )
      {
      this->SpeciesButton->SetParent ( NULL );
      this->SpeciesButton->Delete();
      this->SpeciesButton = NULL;      
      }
    if ( this->SpeciesLabel )
      {
      this->SpeciesLabel->SetParent ( NULL );
      this->SpeciesLabel->Delete();
      this->SpeciesLabel = NULL;
      }
    if ( this->SpeciesNoneButton )
      {
      this->SpeciesNoneButton->SetParent ( NULL );
      this->SpeciesNoneButton->Delete();
      this->SpeciesNoneButton = NULL;      
      }
    if ( this->SpeciesHumanButton )
      {
      this->SpeciesHumanButton->SetParent ( NULL );
      this->SpeciesHumanButton->Delete();
      this->SpeciesHumanButton = NULL;
      }
    if ( this->SpeciesMouseButton )
      {
      this->SpeciesMouseButton->SetParent ( NULL );
      this->SpeciesMouseButton->Delete();
      this->SpeciesMouseButton = NULL;
      }
    if ( this->SpeciesMacaqueButton )
      {
      this->SpeciesMacaqueButton->SetParent ( NULL );
      this->SpeciesMacaqueButton->Delete();
      this->SpeciesMacaqueButton = NULL;
      }
    //---
    // search term structure panel
    //---
    if ( this->StructureFrame )
      {
      this->StructureFrame->SetParent ( NULL );
      this->StructureFrame->Delete();
      this->StructureFrame = NULL;      
      }
    if ( this->StructureMenuButton)
      {
      this->StructureMenuButton->SetParent ( NULL );
      this->StructureMenuButton->Delete();
      this->StructureMenuButton = NULL;
      }
    //---
    // search term structure panel
    //---
    if ( this->StructureListWidget )
      {
      this->StructureListWidget->SetParent ( NULL );
      this->StructureListWidget->Delete ( );
      this->StructureListWidget = NULL;
      }
    //---
    // search term substructure panel
    //---
    if ( this->OtherFrame )
      {
      this->OtherFrame->SetParent ( NULL );
      this->OtherFrame->Delete();
      this->OtherFrame = NULL;      
      }
    if ( this->OtherListWidget )
      {
      this->OtherListWidget->SetParent ( NULL );
      this->OtherListWidget->Delete();
      this->OtherListWidget = NULL;      
      }

#endif

    //---
    // query panel
    //---
#ifdef QUERIES_FRAME
    if ( this->SearchButton )
      {
      this->SearchButton->SetParent ( NULL );
      this->SearchButton->Delete ( );
      this->SearchButton = NULL;
      }
    if ( this->DatabasesMenuButton )
      {
      this->DatabasesMenuButton->SetParent ( NULL );
      this->DatabasesMenuButton->Delete ( );
      this->DatabasesMenuButton = NULL;      
      }
    if ( this->ResultsWithAnyButton )
      {
      this->ResultsWithAnyButton->SetParent ( NULL );
      this->ResultsWithAnyButton->Delete();
      this->ResultsWithAnyButton = NULL;
      }
    if ( this->ResultsWithAllButton )
      {
      this->ResultsWithAllButton->SetParent ( NULL );
      this->ResultsWithAllButton->Delete();
      this->ResultsWithAllButton = NULL;      
      }
    if ( this->ResultsWithExactButton )
      {
      this->ResultsWithExactButton->SetParent ( NULL );
      this->ResultsWithExactButton->Delete();
      this->ResultsWithExactButton = NULL;      
      }
#endif


    //---
    // ontology frame
    //---
#ifdef ONTOLOGY_FRAME
    if ( this->LocalSearchTermEntry )
      {
      this->LocalSearchTermEntry->SetParent ( NULL );
      this->LocalSearchTermEntry->Delete();
      this->LocalSearchTermEntry = NULL;
      }
    if ( this->SynonymsMenuButton )
      {
      this->SynonymsMenuButton->SetParent ( NULL );
      this->SynonymsMenuButton->Delete();
      this->SynonymsMenuButton = NULL;      
      }
    if ( this->BIRNLexEntry )
      {
      this->BIRNLexEntry->SetParent ( NULL );
      this->BIRNLexEntry->Delete();
      this->BIRNLexEntry = NULL;      
      }
    if ( this->BIRNLexIDEntry )
      {
      this->BIRNLexIDEntry->SetParent ( NULL );
      this->BIRNLexIDEntry->Delete();
      this->BIRNLexIDEntry = NULL;      
      }
    if ( this->NeuroNamesEntry )
      {
      this->NeuroNamesEntry->SetParent ( NULL );
      this->NeuroNamesEntry->Delete ( );
      this->NeuroNamesEntry = NULL;      
      }
    if ( this->NeuroNamesIDEntry )
      {
      this->NeuroNamesIDEntry->SetParent ( NULL );
      this->NeuroNamesIDEntry->Delete ( );
      this->NeuroNamesIDEntry = NULL;      
      }
    if ( this->UMLSCIDEntry )
      {
      this->UMLSCIDEntry->SetParent ( NULL );
      this->UMLSCIDEntry->Delete ( );
      this->UMLSCIDEntry = NULL;      
      }
    if ( this->AddLocalTermButton )
      {
      this->AddLocalTermButton->SetParent ( NULL );
      this->AddLocalTermButton->Delete ( );
      this->AddLocalTermButton = NULL;      
      }
    if ( this->AddSynonymButton )
      {
      this->AddSynonymButton->SetParent ( NULL );
      this->AddSynonymButton->Delete ();
      this->AddSynonymButton = NULL;      
      }
    if ( this->AddBIRNLexStringButton )
      {
      this->AddBIRNLexStringButton->SetParent ( NULL );
      this->AddBIRNLexStringButton->Delete ( );
      this->AddBIRNLexStringButton = NULL;      
      }
    if ( this->AddBIRNLexIDButton)
      {
      this->AddBIRNLexIDButton->SetParent ( NULL );
      this->AddBIRNLexIDButton->Delete ( );
      this->AddBIRNLexIDButton = NULL;      
      }
    if ( this->AddNeuroNamesStringButton )
      {
      this->AddNeuroNamesStringButton->SetParent ( NULL );
      this->AddNeuroNamesStringButton->Delete ();
      this->AddNeuroNamesStringButton = NULL;      
      }
    if ( this->AddNeuroNamesIDButton )
      {    //---
      this->AddNeuroNamesIDButton->SetParent ( NULL );
      this->AddNeuroNamesIDButton->Delete ( );
      this->AddNeuroNamesIDButton = NULL;      
      }
    if ( this->AddUMLSCIDButton )
      {
      this->AddUMLSCIDButton->SetParent ( NULL );
      this->AddUMLSCIDButton->Delete ();
      this->AddUMLSCIDButton = NULL;      
      }
    if ( this->NeuroNamesHierarchyButton)
      {
      this->NeuroNamesHierarchyButton->SetParent ( NULL );
      this->NeuroNamesHierarchyButton->Delete();
      this->NeuroNamesHierarchyButton = NULL;      
      }
    if ( this->BIRNLexHierarchyButton )
      {
      this->BIRNLexHierarchyButton->SetParent ( NULL );
      this->BIRNLexHierarchyButton->Delete();
      this->BIRNLexHierarchyButton= NULL;
      }
    if ( this->UMLSHierarchyButton )
      {
      this->UMLSHierarchyButton->SetParent ( NULL );
      this->UMLSHierarchyButton->Delete();
      this->UMLSHierarchyButton= NULL;
      }
    if ( this->SavedTerms )
      {
      this->SavedTerms->SetParent ( NULL );
      this->SavedTerms->Delete();
      this->SavedTerms = NULL;
      }
#endif


    //---
    // results panel
    //---
#ifdef RESULTS_FRAME
    if ( this->CurrentResultsList )
      {
      this->CurrentResultsList->SetParent(NULL);
      this->CurrentResultsList->Delete();
      this->CurrentResultsList = NULL;
      }
    if ( this->AccumulatedResultsList )
      {
      this->AccumulatedResultsList->SetParent(NULL);
      this->AccumulatedResultsList->Delete();
      this->AccumulatedResultsList = NULL;
      }
    if ( this->DeselectAllCurrentResultsButton )
      {
      this->DeselectAllCurrentResultsButton->SetParent ( NULL );      
      this->DeselectAllCurrentResultsButton->Delete();
      this->DeselectAllCurrentResultsButton = NULL;      
      }
    if ( this->DeselectAllAccumulatedResultsButton )
      {
      this->DeselectAllAccumulatedResultsButton->SetParent ( NULL );      
      this->DeselectAllAccumulatedResultsButton->Delete();
      this->DeselectAllAccumulatedResultsButton = NULL;      
      }
    if ( this->DeleteCurrentResultButton )
      {
      this->DeleteCurrentResultButton->SetParent(NULL);
      this->DeleteCurrentResultButton->Delete();
      this->DeleteCurrentResultButton = NULL;
      }
    if ( this->DeleteAllCurrentResultsButton )
      {
      this->DeleteAllCurrentResultsButton->SetParent(NULL);
      this->DeleteAllCurrentResultsButton->Delete();
      this->DeleteAllCurrentResultsButton = NULL;
      }
    if ( this->DeleteAccumulatedResultButton )
      {
      this->DeleteAccumulatedResultButton->SetParent(NULL);
      this->DeleteAccumulatedResultButton->Delete();
      this->DeleteAccumulatedResultButton = NULL;
      }
    if ( this->DeleteAllAccumulatedResultsButton )
      {
      this->DeleteAllAccumulatedResultsButton->SetParent(NULL);
      this->DeleteAllAccumulatedResultsButton->Delete();
      this->DeleteAllAccumulatedResultsButton = NULL;
      }
    if ( this->SaveCurrentResultsButton )
      {
      this->SaveCurrentResultsButton->SetParent(NULL);
      this->SaveCurrentResultsButton->Delete();
      this->SaveCurrentResultsButton = NULL;
      }
    if ( this->SaveCurrentSelectedResultsButton )
      {
      this->SaveCurrentSelectedResultsButton->SetParent(NULL);
      this->SaveCurrentSelectedResultsButton->Delete();
      this->SaveCurrentSelectedResultsButton = NULL;
      }
    if ( this->SaveAccumulatedResultsButton )
      {
      this->SaveAccumulatedResultsButton->SetParent(NULL);
      this->SaveAccumulatedResultsButton->Delete();
      this->SaveAccumulatedResultsButton = NULL;
      }
    if ( this->LoadURIsButton )
      {
      this->LoadURIsButton->SetParent ( NULL );
      this->LoadURIsButton->Delete();
      this->LoadURIsButton = NULL;
      }
#endif
    
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::OpenBIRNLexBrowser()
{
  this->Script ( "QueryAtlasLaunchOntologyBrowser BIRN" );
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::OpenNeuroNamesBrowser()
{
  this->Script ( "QueryAtlasLaunchOntologyBrowser NN" );
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::OpenUMLSBrowser()
{
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "QueryAtlasGUI: " << this->GetClassName ( ) << "\n";
    os << indent << "Logic: " << this->GetLogic ( ) << "\n";

    //---
    // load frame
    //---
#ifdef LOAD_FRAME
    os << indent << "FSasegSelector: " << this->GetFSasegSelector ( ) << "\n";    
    os << indent << "FSmodelSelector: " << this->GetFSmodelSelector ( ) << "\n";    
    os << indent << "FSbrainSelector: " << this->GetFSbrainSelector ( ) << "\n";    
    os << indent << "FSstatsSelector: " << this->GetFSstatsSelector ( ) << "\n";    
    os << indent << "FSgoButton: " << this->GetFSgoButton() << "\n";
    os << indent << "QdecGoButton: " << this->GetQdecGoButton() << "\n";
    os << indent << "QdecGetResultsButton: " << this->GetQdecGetResultsButton ( ) << "\n";    
    os << indent << "QdecScalarSelector: " << this->GetQdecScalarSelector ( ) << "\n";    
    os << indent << "QdecModelSelector: " << this->GetQdecModelSelector ( ) << "\n";    
#endif
    
    //---
    // ontology frame
    //---
#ifdef ONTOLOGY_FRAME
    os << indent << "LoadSearchTermEntry" << this->GetLocalSearchTermEntry ( ) << "\n";    
    os << indent << "SynonymsMenuButton" << this->GetSynonymsMenuButton ( ) << "\n";    
    os << indent << "BIRNLexEntry" << this->GetBIRNLexEntry ( ) << "\n";    
    os << indent << "BIRNLexIDEntry" << this->GetBIRNLexIDEntry ( ) << "\n";    
    os << indent << "NeuroNamesEntry" << this->GetNeuroNamesEntry ( ) << "\n";    
    os << indent << "NeuroNamesIDEntry" << this->GetNeuroNamesIDEntry ( ) << "\n";    
    os << indent << "UMLSCIDEntry" << this->GetUMLSCIDEntry ( ) << "\n";    
    os << indent << "AddLocalTermButton" << this->GetAddLocalTermButton ( ) << "\n";    
    os << indent << "AddSynonymButton" << this->GetAddSynonymButton ( ) << "\n";    
    os << indent << "AddBIRNLexStringButton" << this->GetAddBIRNLexStringButton ( ) << "\n";    
    os << indent << "AddBIRNLexIDButton" << this->GetAddBIRNLexIDButton ( ) << "\n";    
    os << indent << "AddNeuroNamesStringButton" << this->GetAddNeuroNamesStringButton ( ) << "\n";    
    os << indent << "AddNeuroNamesIDButton" << this->GetAddNeuroNamesIDButton ( ) << "\n";    
    os << indent << "AddUMLSCIDButton" << this->GetAddUMLSCIDButton ( ) << "\n";    
    os << indent << "BIRNLexHierarchyButton" << this->GetBIRNLexHierarchyButton ( ) << "\n";    
    os << indent << "NeuroNamesHierarchyButton" << this->GetNeuroNamesHierarchyButton ( ) << "\n";    
    os << indent << "UMLSHierarchyButton" << this->GetUMLSHierarchyButton ( ) << "\n";    
    os << indent << "SavedTerms" << this->GetSavedTerms() << "\n";
    os << indent << "AddDiagnosisEntry" << this->GetAddDiagnosisEntry() << "\n";
    os << indent << "ResultsWithExactButton" <<  this->GetResultsWithExactButton() << "\n";
    os << indent << "ResultsWithAnyButton" <<  this->GetResultsWithAnyButton() << "\n";
    os << indent << "ResultsWithAllButton" <<  this->GetResultsWithAllButton() << "\n";
#endif
    
    //---
    // TODO: finish this method!
    //---
}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::RemoveGUIObservers ( )
{
  vtkDebugMacro("vtkQueryAtlasGUI: RemoveGUIObservers\n");

  this->FIPSFSButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->QdecButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->FSasegSelector->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->FSbrainSelector->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->FSstatsSelector->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->FSgoButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->QdecGoButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->FSmodelSelector->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->QdecScalarSelector->GetWidget()->GetMenu()->RemoveObservers ( vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->QdecModelSelector->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  
  this->AddLocalTermButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddSynonymButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddBIRNLexStringButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddBIRNLexIDButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddNeuroNamesStringButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddNeuroNamesIDButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddUMLSCIDButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->SavedTerms->RemoveWidgetObservers();
  this->SavedTerms->RemoveObservers(vtkQueryAtlasSearchTermWidget::ReservedTermsEvent, (vtkCommand *)this->GUICallbackCommand );  

  this->StructureButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->StructureListWidget->RemoveWidgetObservers();
  this->OtherButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OtherListWidget->RemoveWidgetObservers();
  this->SpeciesButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SpeciesNoneButton->RemoveObservers(vtkKWRadioButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SpeciesHumanButton->RemoveObservers(vtkKWRadioButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SpeciesMouseButton->RemoveObservers(vtkKWRadioButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SpeciesMacaqueButton->RemoveObservers(vtkKWRadioButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->ModelVisibilityButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->AnnotationVisibilityButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->AnnotationTermSetMenuButton->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->BIRNLexHierarchyButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->NeuroNamesHierarchyButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->BIRNLexHierarchyButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->LocalSearchTermEntry->RemoveObservers(vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->PopulationButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DiagnosisMenuButton->GetWidget()->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->HandednessMenuButton->GetWidget()->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->GenderMenuButton->GetWidget()->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->AgeMenuButton->GetWidget()->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->AddDiagnosisEntry->GetWidget()->RemoveObservers ( vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->SearchButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

#ifdef RESULTS_FRAME
  this->DeselectAllCurrentResultsButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DeselectAllAccumulatedResultsButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SaveCurrentResultsButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SaveCurrentSelectedResultsButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
//  this->SaveAccumulatedResultsButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
//  this->LoadURIsButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DeleteAccumulatedResultButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DeleteAllAccumulatedResultsButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DeleteCurrentResultButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DeleteAllCurrentResultsButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->CurrentResultsList->GetWidget()->RemoveObservers(vtkKWListBox::ListBoxSelectionChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  //  this->AccumulatedResultsList->GetWidget()->RemoveObservers(vtkKWListBox::ListBoxSelectionChangedEvent, (vtkCommand *)this->GUICallbackCommand );
#endif
  
/*
 //--MRML
  if (this->MRMLScene)
      {
        this->MRMLScene->RemoveObservers(vtkMRMLScene::NodeRemovedEvent, (vtkCommand *)this->GUICallbackCommand);
        this->MRMLScene->RemoveObservers(vtkMRMLScene::NodeAddedEvent, (vtkCommand *)this->GUICallbackCommand);
        this->MRMLScene->RemoveObservers(vtkMRMLScene::SceneCloseEvent, (vtkCommand *)this->GUICallbackCommand);
      }
*/

}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::AddGUIObservers ( )
{
  vtkDebugMacro("vtkQueryAtlasGUI: AddGUIObservers\n");
  this->FIPSFSButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->QdecButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->FSasegSelector->AddObserver ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->FSbrainSelector->AddObserver ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->FSstatsSelector->AddObserver ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->FSmodelSelector->AddObserver ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->FSgoButton->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->QdecGoButton->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->QdecScalarSelector->GetWidget()->GetMenu()->AddObserver ( vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->QdecModelSelector->AddObserver ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  

  this->AddLocalTermButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddSynonymButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddBIRNLexStringButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddBIRNLexIDButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddNeuroNamesStringButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddNeuroNamesIDButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->AddUMLSCIDButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->SavedTerms->AddWidgetObservers();
  this->SavedTerms->AddObserver(vtkQueryAtlasSearchTermWidget::ReservedTermsEvent, (vtkCommand *)this->GUICallbackCommand );  

  this->StructureButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->StructureListWidget->AddWidgetObservers();
  this->OtherButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OtherListWidget->AddWidgetObservers();
  this->SpeciesButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SpeciesNoneButton->AddObserver(vtkKWRadioButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SpeciesHumanButton->AddObserver(vtkKWRadioButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SpeciesMouseButton->AddObserver(vtkKWRadioButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SpeciesMacaqueButton->AddObserver(vtkKWRadioButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->ModelVisibilityButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->AnnotationVisibilityButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->AnnotationTermSetMenuButton->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  
  this->BIRNLexHierarchyButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->NeuroNamesHierarchyButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->BIRNLexHierarchyButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->LocalSearchTermEntry->AddObserver(vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->PopulationButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DiagnosisMenuButton->GetWidget()->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->HandednessMenuButton->GetWidget()->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->GenderMenuButton->GetWidget()->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->AgeMenuButton->GetWidget()->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->AddDiagnosisEntry->GetWidget()->AddObserver(vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->SearchButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  
#ifdef RESULTS_FRAME
  this->DeselectAllCurrentResultsButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DeselectAllAccumulatedResultsButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SaveCurrentResultsButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SaveCurrentSelectedResultsButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
//  this->SaveAccumulatedResultsButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
//  this->LoadURIsButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DeleteAccumulatedResultButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DeleteAllAccumulatedResultsButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DeleteCurrentResultButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->DeleteAllCurrentResultsButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->CurrentResultsList->GetWidget()->AddObserver(vtkKWListBox::ListBoxSelectionChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  //  this->AccumulatedResultsList->GetWidget()->AddObserver(vtkKWListBox::ListBoxSelectionChangedEvent, (vtkCommand *)this->GUICallbackCommand );
#endif
  

/*
 //--MRML
  if (this->MRMLScene)
      {
      if (this->MRMLScene->HasObserver(vtkMRMLScene::NodeRemovedEvent, (vtkCommand *)this->GUICallbackCommand) != 1)
        {
        this->MRMLScene->AddObserver(vtkMRMLScene::NodeRemovedEvent, (vtkCommand *)this->GUICallbackCommand);
        }
      if (this->MRMLScene->HasObserver(vtkMRMLScene::NodeAddedEvent, (vtkCommand *)this->GUICallbackCommand) != 1)
        {
        this->MRMLScene->AddObserver(vtkMRMLScene::NodeAddedEvent, (vtkCommand *)this->GUICallbackCommand);
        }
      if (this->MRMLScene->HasObserver(vtkMRMLScene::SceneCloseEvent, (vtkCommand *)this->GUICallbackCommand) != 1)
        {
        this->MRMLScene->AddObserver(vtkMRMLScene::SceneCloseEvent, (vtkCommand *)this->GUICallbackCommand);
        }
      }
*/
}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::ProcessGUIEvents ( vtkObject *caller,
                                            unsigned long event, void *callData )
{
    // nothing to do here yet...
  vtkKWPushButton *b = vtkKWPushButton::SafeDownCast ( caller );
  vtkKWMenu *m = vtkKWMenu::SafeDownCast ( caller );
  vtkKWCheckButton *c = vtkKWCheckButton::SafeDownCast ( caller );
  vtkKWListBox *lb = vtkKWListBox::SafeDownCast ( caller );
  vtkKWEntry *e  = vtkKWEntry::SafeDownCast ( caller);
  vtkSlicerNodeSelectorWidget *sel = vtkSlicerNodeSelectorWidget::SafeDownCast ( caller );
  vtkQueryAtlasSearchTermWidget *stw = vtkQueryAtlasSearchTermWidget::SafeDownCast (caller );

  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  vtkMRMLNode *node;
  
  if ( (stw = this->SavedTerms) && (event == vtkQueryAtlasSearchTermWidget::ReservedTermsEvent ))
    {
    // test
    int num = this->SavedTerms->GetNumberOfReservedTerms();
    const char *term;
    for (int i=0; i<num; i++)
      {
      term = this->SavedTerms->GetNthReservedTerm ( i );
      this->StructureListWidget->AddNewSearchTerm ( term );
      }
    }

/*
  //MRML
  if (vtkMRMLScene::SafeDownCast(caller) != NULL &&
      vtkMRMLScene::SafeDownCast(caller) == this->MRMLScene &&
      event == vtkMRMLScene::NodeRemovedEvent)
    {
    // check to see if the model or labels have been deleted.
    //--- check to see if the lh.pial has been deleted
    //--- and clean up if so.
    vtkMRMLModelNode *node;
    int n = this->MRMLScene->GetNumberOfNodesByClass( "vtkMRMLModelNode");
    for ( int i=0; i < n; i++ )
      {
      node = vtkMRMLModelNode::SafeDownCast ( this->MRMLScene->GetNthNodeByClass ( i, "vtkMRMLModelNode") );
      if ( (!strcmp ( node->GetName(), "lh.pial")) ||  (! strcmp ( node->GetName(), "lh.inflated")) )
        {
        // ok, query model for either qdec or fips/freesurfer is still here;
        // no op
        }
      else
        {
        this->Script ( "QueryAtlasTearDown; QueryAtlasInitializeGlobasl");
        this->Script ("QueryAtlasNodeRemovedUpdate" );
        break;
        }
      }
    }
  if (vtkMRMLScene::SafeDownCast(caller) != NULL &&
      vtkMRMLScene::SafeDownCast(caller) == this->MRMLScene &&
      event == vtkMRMLScene::NodeAddedEvent)
    {
    this->Script ( "QueryAtlasNodeAddedUpdate" );
    }
  if (vtkMRMLScene::SafeDownCast(caller) != NULL &&
      vtkMRMLScene::SafeDownCast(caller) == this->MRMLScene &&
      event == vtkMRMLScene::SceneCloseEvent)
    {
    //
    }
*/
  
  if ((sel == this->FSasegSelector ) && ( event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent ) )
    {
    node = sel->GetSelected();
    if ( node != NULL )
      {
      this->Script ( "QueryAtlasSetAnnotatedLabelMap" );
      }
    }
  else if ((sel == this->FSmodelSelector ) && ( event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent ) )
    {
    node = sel->GetSelected();
    if ( node != NULL )
      {
      this->Script ( "QueryAtlasSetAnnotatedModel" );
      }
    }
  else if ((sel == this->FSbrainSelector ) && ( event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent ) )
    {
    node = sel->GetSelected();
    if ( node != NULL )
      {
      this->Script ( "QueryAtlasSetAnatomical" );
      }
    }
  else if ((sel == this->FSstatsSelector ) && ( event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent ) )
    {
    node = sel->GetSelected();
    if ( node != NULL )
      {
      this->Script ( "QueryAtlasSetStatistics" );
      }
    }

  
  //---
  //--- Process All Entry events
  //---
  if ( this->AddDiagnosisEntry )
    {
    if ( (e == this->AddDiagnosisEntry->GetWidget() ) && ( event == vtkKWEntry::EntryValueChangedEvent ))
      {
      if ( strcmp (this->AddDiagnosisEntry->GetWidget()->GetValue(), "" ) ) 
        {
        this->AddToDiagnosisMenu ( this->DiagnosisMenuButton->GetWidget()->GetMenu(),
                                   this->AddDiagnosisEntry->GetWidget()->GetValue() );
        }
      }
    }

  if ( (e == this->LocalSearchTermEntry) && (event == vtkKWEntry::EntryValueChangedEvent) )
    {
    if ( this->LocalSearchTermEntry->GetValue() )
      {
      if ( strcmp ( this->LocalSearchTermEntry->GetValue(), "" )) 
        {
        this->Script ("QueryAtlasPopulateOntologyInformation %s local", this->LocalSearchTermEntry->GetValue() );
        }
      }
    }
  else if ( (e == this->BIRNLexEntry) && (event == vtkKWEntry::EntryValueChangedEvent) )
    {
    if (this->BIRNLexEntry->GetValue() )
      {
      if ( strcmp ( this->BIRNLexEntry->GetValue(), "" ))
        {
        this->Script ("QueryAtlasPopulateOntologyInformation %s BIRN_String", this->BIRNLexEntry->GetValue() );
        }
      }
    }
  else if ( (e == this->BIRNLexIDEntry) && (event == vtkKWEntry::EntryValueChangedEvent) )
    {
    if ( this->BIRNLexIDEntry->GetValue() )
      {
      if ( strcmp (this->BIRNLexIDEntry->GetValue(), "" ))
        {
        this->Script ("QueryAtlasPopulateOntologyInformation %s BIRN_ID", this->BIRNLexIDEntry->GetValue() );
        }
      }
    }
  else if ( (e == this->NeuroNamesEntry) && (event == vtkKWEntry::EntryValueChangedEvent) )
    {
    if (this->NeuroNamesEntry->GetValue() )
      {
      if  (strcmp (this->NeuroNamesEntry->GetValue(), "" ))
        {
        this->Script ("QueryAtlasPopulateOntologyInformation %s NN", this->NeuroNamesEntry->GetValue() );
        }
      }
    }
  else if ( (e == this->NeuroNamesIDEntry) && (event == vtkKWEntry::EntryValueChangedEvent) )
    {
    if ( this->NeuroNamesIDEntry->GetValue() )
      {
      if ( strcmp (this->NeuroNamesIDEntry->GetValue(), "" ))
        {
        this->Script ("QueryAtlasPopulateOntologyInformation %s NN_ID", this->NeuroNamesIDEntry->GetValue() );
        }
      }
    }
  else if ( (e == this->UMLSCIDEntry) && (event == vtkKWEntry::EntryValueChangedEvent) )
    {
    if ( this->UMLSCIDEntry->GetValue() )
      {
      if ( strcmp (this->UMLSCIDEntry->GetValue(), "" ))
        {
        this->Script ("QueryAtlasPopulateOntologyInformation %s UMLS_CID", this->UMLSCIDEntry->GetValue() );
        }
      }
    }

  //---
  //--- Process All PushButton events
  //---
  if ( (b == this->NeuroNamesHierarchyButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    }
  else if ( (b == this->FSgoButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->Script ( "QueryAtlasFipsFreeSurferSetUp" );
    }
  else if ( (b == this->QdecGoButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->Script ( "QueryAtlasQdecSetUp");
    }
  else if ( (b == this->BIRNLexHierarchyButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    //--- TODO: check to see if BIRNLexBrowser is open.
    
    //--- Will open if it's not alreay open.
    this->OpenBIRNLexBrowser();
    //--- get last clicked (or typed) structure from the LocalSearchTermEntry
    const char *structureLabel =  this->LocalSearchTermEntry->GetValue();
    if ( !strcmp (structureLabel, "" ))
      {
      structureLabel = "BIRNLex_subset";
      }
    this->Script ( "QueryAtlasSendHierarchyCommand  %s", structureLabel );
    }
  else if ( (b == this->AddLocalTermButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->SavedTerms->AddTerm (this->LocalSearchTermEntry->GetValue() );
    }
  else if ( (b == this->AddSynonymButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->SavedTerms->AddTerm (this->SynonymsMenuButton->GetValue() );
    }
  else if ( (b == this->AddBIRNLexStringButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->SavedTerms->AddTerm (this->BIRNLexEntry->GetValue() );
    }
  else if ( (b == this->AddBIRNLexIDButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->SavedTerms->AddTerm (this->BIRNLexIDEntry->GetValue() );
    }
  else if ( (b == this->AddNeuroNamesStringButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->SavedTerms->AddTerm (this->NeuroNamesEntry->GetValue() );
    }
  else if ( (b == this->AddNeuroNamesIDButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->SavedTerms->AddTerm (this->NeuroNamesIDEntry->GetValue() );
    }
  else if ( (b == this->AddUMLSCIDButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->SavedTerms->AddTerm (this->UMLSCIDEntry->GetValue() );
    }
  else if ( (b == this->SearchButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->Script ( "QueryAtlasFormURLsForTargets");
    }
  else if ( (b == this->AnnotationVisibilityButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    if ( this->AnnotationVisibility == 1 )
      {
      // turn off automatic annotations in the main viewer
      vtkKWIcon *i = app->GetApplicationGUI()->GetMainSliceGUI0()->GetSliceController()->GetVisibilityIcons()->GetInvisibleIcon();
      this->AnnotationVisibilityButton->SetImageToIcon ( i );
      this->AnnotationVisibility = 0;
      this->Script ( "QueryAtlasSetAnnotationsInvisible" );
      }
    else
      {
      // turn on automatic annotations in main viewer
      vtkKWIcon *i = app->GetApplicationGUI()->GetMainSliceGUI0()->GetSliceController()->GetVisibilityIcons()->GetVisibleIcon();
      this->AnnotationVisibilityButton->SetImageToIcon ( i );
      this->AnnotationVisibility = 1;
     this->Script ( "QueryAtlasSetAnnotationsVisible" );
      }
    }
  else if ( (b == this->ModelVisibilityButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    if ( this->ModelVisibility == 1 )
      {
      // turn off automatic annotations in the main viewer
      vtkKWIcon *i = app->GetApplicationGUI()->GetMainSliceGUI0()->GetSliceController()->GetVisibilityIcons()->GetInvisibleIcon();
      this->ModelVisibilityButton->SetImageToIcon ( i );
      this->ModelVisibility = 0;
      this->Script ( "QueryAtlasSetQueryModelInvisible" );
      }
    else
      {
      // turn on automatic annotations in main viewer
      vtkKWIcon *i = app->GetApplicationGUI()->GetMainSliceGUI0()->GetSliceController()->GetVisibilityIcons()->GetVisibleIcon();
      this->ModelVisibilityButton->SetImageToIcon ( i );
      this->ModelVisibility = 1;
      this->Script ( "QueryAtlasSetQueryModelVisible" );
      }
    }
  else if ( (b == this->FIPSFSButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->UnpackLoaderContextFrames();
    this->PackLoaderContextFrame ( this->FIPSFSFrame );
    this->ColorCodeLoaderContextButtons ( this->FIPSFSButton );
    }
  else if ( (b == this->QdecButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->UnpackLoaderContextFrames();
    this->PackLoaderContextFrame ( this->QdecFrame );
    this->ColorCodeLoaderContextButtons ( this->QdecButton );
    }
  else if ( (b == this->StructureButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->UnpackQueryBuilderContextFrames();
    this->PackQueryBuilderContextFrame ( this->StructureFrame );
    this->ColorCodeContextButtons ( this->StructureButton );
    }
  else if ( (b == this->OtherButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->UnpackQueryBuilderContextFrames();
    this->PackQueryBuilderContextFrame ( this->OtherFrame);    
    this->ColorCodeContextButtons ( this->OtherButton );
    }
  else if ( (b == this->PopulationButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->UnpackQueryBuilderContextFrames();
    this->PackQueryBuilderContextFrame ( this->PopulationFrame );
    this->ColorCodeContextButtons ( this->PopulationButton );
    }
  else if ( (b == this->SpeciesButton) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->UnpackQueryBuilderContextFrames();
    this->PackQueryBuilderContextFrame ( this->SpeciesFrame );
    this->ColorCodeContextButtons ( this->SpeciesButton );
    }
  else if ( (b == this->DeselectAllCurrentResultsButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    int num = this->CurrentResultsList->GetWidget()->GetNumberOfItems();
    for ( int i=0; i<num; i++ )
      {
      this->CurrentResultsList->GetWidget()->SetSelectState(i,0);
      }
    }
  else if ( (b == this->DeleteCurrentResultButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    int num = this->CurrentResultsList->GetWidget()->GetNumberOfItems();
    for ( int i=0; i<num; i++ )
      {
      if ( this->CurrentResultsList->GetWidget()->GetSelectState(i) )
        {
        this->CurrentResultsList->GetWidget()->DeleteRange( i,i );
        }
      }
    }
  else if ( (b == this->DeleteAllCurrentResultsButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->CurrentResultsList->GetWidget()->DeleteAll();
    }
  else if ( (b == this->SaveCurrentResultsButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    int num = this->CurrentResultsList->GetWidget()->GetNumberOfItems();
    for ( int i=0; i<num; i++ )
      {
      this->AccumulatedResultsList->GetWidget()->AppendUnique (this->CurrentResultsList->GetWidget()->GetItem( i ) );
      }
    }
  else if ( (b == this->SaveCurrentSelectedResultsButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    int num = this->CurrentResultsList->GetWidget()->GetNumberOfItems();
    for ( int i=0; i<num; i++ )
      {
      if ( this->CurrentResultsList->GetWidget()->GetSelectState(i) )
        {
        this->AccumulatedResultsList->GetWidget()->AppendUnique (this->CurrentResultsList->GetWidget()->GetItem( i ) );
        }
      }
    }


  else if ( (b == this->DeselectAllAccumulatedResultsButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    int num = this->AccumulatedResultsList->GetWidget()->GetNumberOfItems();
    for ( int i=0; i<num; i++ )
      {
      this->AccumulatedResultsList->GetWidget()->SetSelectState(i,0);
      }
    }
  else if ( (b == this->DeleteAccumulatedResultButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    int num = this->AccumulatedResultsList->GetWidget()->GetNumberOfItems();
    for ( int i=0; i<num; i++ )
      {
      if ( this->AccumulatedResultsList->GetWidget()->GetSelectState(i) )
        {
        this->AccumulatedResultsList->GetWidget()->DeleteRange( i,i );
        }
      }
    }
  else if ( (b == this->DeleteAllAccumulatedResultsButton ) && (event == vtkKWPushButton::InvokedEvent ) )
    {
    this->AccumulatedResultsList->GetWidget()->DeleteAll();
    }
//  else if ( (b == this->SaveAccumulatedResultsButton ) && (event == vtkKWPushButton::InvokedEvent ) )
//    {
//    this->Script( "QueryAtlasWriteFirefoxBookmarkFile");
//    }
//  else if ( (b == this->LoadURIsButton ) && (event == vtkKWPushButton::InvokedEvent ) )
//    {
//    this->Script( "QueryAtlasLoadFirefoxBookmarkFile");
//    }


  //---
  //--- Process menu selections
  //---
  // no need to do anything here; we'll just grab the widget values when we need them with tcl
  if ( this->AnnotationTermSetMenuButton )
    {
    if (( m== this->AnnotationTermSetMenuButton->GetMenu()) && (event == vtkKWMenu::MenuItemInvokedEvent ) )
      {
      const char *val = this->AnnotationTermSetMenuButton->GetValue();
      if ( !strcmp( val, "local identifier" ) )
        {
        this->Script ( "QueryAtlasSetAnnotationTermSet local" );
        }
      else if (!strcmp( val, "BIRNLex String" ) )
        {
        this->Script ( "QueryAtlasSetAnnotationTermSet BIRNLex" );
        }
      else if (!strcmp( val, "NeuroNames String" ) )
        {
        this->Script ( "QueryAtlasSetAnnotationTermSet NeuroNames" );
        }
      else if (!strcmp( val, "UMLS CID" ) )
        {
        this->Script ( "QueryAtlasSetAnnotationTermSet UMLS" );
        }
      }
    }
  if ( this->DiagnosisMenuButton )
    {
    if (( m== this->DiagnosisMenuButton->GetWidget()->GetMenu()) && (event == vtkKWMenu::MenuItemInvokedEvent ) )
      {
      }
    }
  if ( this->GenderMenuButton )
    {
    if (( m== this->GenderMenuButton->GetWidget()->GetMenu()) && (event == vtkKWMenu::MenuItemInvokedEvent ) )
      {
      }
    }
  if ( this->HandednessMenuButton )
    {
    if (( m== this->HandednessMenuButton->GetWidget()->GetMenu()) && (event == vtkKWMenu::MenuItemInvokedEvent ) )
      {
      }
    }
  if ( this->AgeMenuButton )
    {
    if (( m== this->AgeMenuButton->GetWidget()->GetMenu()) && (event == vtkKWMenu::MenuItemInvokedEvent ) )
      {
      }
    }
  
  if ((c == this->SpeciesNoneButton) && (event == vtkKWRadioButton::SelectedStateChangedEvent))
    {
    }
  if ((c == this->SpeciesHumanButton) && (event == vtkKWRadioButton::SelectedStateChangedEvent))
    {
    }
  if ((c == this->SpeciesMouseButton) && (event == vtkKWRadioButton::SelectedStateChangedEvent))
    {
    }
  if ((c == this->SpeciesMacaqueButton) && (event == vtkKWRadioButton::SelectedStateChangedEvent))
    {
    }
    return;
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::WriteBookmarksCallback ()
{
  // get file from dialog
  const char *filen;
  
  filen = this->SaveAccumulatedResultsButton->GetLoadSaveDialog()->GetFileName();
  if ( filen != NULL )
    {
    itksys::SystemTools::ConvertToUnixOutputPath( filen );
    this->Script( "QueryAtlasWriteFirefoxBookmarkFile \"%s\"", filen );
    }
    this->SaveAccumulatedResultsButton->SetText ( "" );
}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::LoadBookmarksCallback ()
{
  // get file from dialog
  const char *filen;
  filen = this->LoadURIsButton->GetLoadSaveDialog()->GetFileName();
  if ( filen != NULL )
    {
    itksys::SystemTools::ConvertToUnixOutputPath( filen );
    this->Script( "QueryAtlasLoadFirefoxBookmarkFile \"%s\"", filen );
    this->LoadURIsButton->SetText ( "" );
    }
}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::ColorCodeContextButtons ( vtkKWPushButton *b )
{
#ifdef SEARCHTERM_FRAME
  this->OtherButton->SetBackgroundColor ( _br, _bg, _bb );
  this->StructureButton->SetBackgroundColor ( _br, _bg, _bb );
  this->PopulationButton->SetBackgroundColor ( _br, _bg, _bb );
  this->SpeciesButton->SetBackgroundColor ( _br, _bg, _bb );

  this->OtherButton->SetForegroundColor ( _fr, _fg, _fb );
  this->StructureButton->SetForegroundColor ( _fr, _fg, _fb );
  this->PopulationButton->SetForegroundColor ( _fr, _fg, _fb );
  this->SpeciesButton->SetForegroundColor ( _fr, _fg, _fb );

  b->SetBackgroundColor (1.0, 1.0, 1.0);
  b->SetForegroundColor (0.0, 0.0, 0.0);
#endif
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::ProcessLogicEvents ( vtkObject *caller,
                                              unsigned long event, void *callData )
{
    // Fill in
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::ProcessMRMLEvents ( vtkObject *caller,
                                             unsigned long event, void *callData )
{    
 if (this->ProcessingMRMLEvent != 0 )
    {
    return;
    }
  this->ProcessingMRMLEvent = event;
  vtkDebugMacro("processing event " << event);
   

  //--- has a node been added?
  if ( vtkMRMLScene::SafeDownCast(caller) == this->MRMLScene 
       && (event == vtkMRMLScene::NodeAddedEvent ) )
    {
    this->Script ( "QueryAtlasNodeAddedUpdate" );
    }

  //--- has a node been deleted?
  if ( vtkMRMLScene::SafeDownCast(caller) == this->MRMLScene 
       && (event == vtkMRMLScene::NodeRemovedEvent ) )
    {
    //this->UpdateFromMRML();
    // check to see if the model or labels have been deleted.
    //--- check to see if the lh.pial has been deleted
    //--- and clean up if so.
    vtkMRMLModelNode *node;
    int n = this->MRMLScene->GetNumberOfNodesByClass( "vtkMRMLModelNode");
    for ( int i=0; i < n; i++ )
      {
      node = vtkMRMLModelNode::SafeDownCast ( this->MRMLScene->GetNthNodeByClass ( i, "vtkMRMLModelNode") );
      if ( (!strcmp ( node->GetName(), "lh.pial")) ||  (! strcmp ( node->GetName(), "lh.inflated")) )
        {
        //ok, query model is still here; no op
        }
      else
        {
        this->Script ( "QueryAtlasTearDown; QueryAtlasInitializeGlobasl");
        this->Script ("QueryAtlasNodeRemovedUpdate" );
        break;
        }
      }
    }
  
  //--- is the scene closing?
  if (event == vtkMRMLScene::SceneCloseEvent )
    {
    this->SceneClosing = true;
    // reset globals.
    this->Script ("QueryAtlasTearDown" );
    this->Script("QueryAtlasInitializeGlobals");
    }
  else 
    {
    this->SceneClosing = false;
    }
  this->ProcessingMRMLEvent = 0;
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::Enter ( )
{
    vtkDebugMacro("vtkQueryAtlasGUI: Enter\n");
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::Exit ( )
{
    vtkDebugMacro("vtkQueryAtlasGUI: Exit\n");
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::AddMRMLObservers()
{
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::RemoveMRMLObservers()
{
  this->SetAndObserveMRMLScene ( NULL );
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildGUI ( )
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  // Define your help text here.


    
    // ---
    // MODULE GUI FRAME 
    // configure a page for a model loading UI for now.
    // later, switch on the modulesButton in the SlicerControlGUI
    // ---
    // create a page
    this->UIPanel->AddPage ( "QueryAtlas", "QueryAtlas", NULL );

    const char *help = "The (Generation 1) Query Atlas module allows interactive Google, Wikipedia, queries from within the 3D anatomical display.";
    const char *about = "This research was supported by Grant 5 MOI RR 000827 to the FIRST BIRN and Grant 1 U24 RR021992 to the FBIRN Biomedical Informatics Research Network (BIRN, http://www.nbirn.net), that is funded by the National Center for Research Resources (NCRR) at the National Institutes of Health (NIH). This work was also supported by NA-MIC, NAC, NCIGT. NeuroNames ontology and URI resources are provided courtesy of BrainInfo, Neuroscience Division, National Primate Research Center, University of Washington (http://www.braininfo.org).                                                                                                                                                                                      ";
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "QueryAtlas" );
    this->QueryAtlasIcons = vtkQueryAtlasIcons::New();
    this->BuildHelpAndAboutFrame ( page, help, about );
    this->BuildAcknowledgementPanel ( );
#ifdef LOAD_FRAME
    this->BuildLoadAndConvertGUI ( );
#endif
#ifdef ANNO_FRAME
    this->BuildAnnotationOptionsGUI ( );
#endif
#ifdef ONTOLOGY_FRAME
    this->BuildOntologyGUI ( );
#endif
#ifdef SEARCHTERM_FRAME
    this->BuildSearchTermGUI ( );
#endif
#ifdef QUERIES_FRAME
    this->BuildQueriesGUI ( );
#endif
//    this->BuildDisplayAndNavigationGUI ( );
      /*
    // ---
    // Source main tcl files.
    // (QueryAtlasInit sources other required tcl files)
    // ---
     vtksys_stl::string slicerHome;
    if (!vtksys::SystemTools::GetEnv("SLICER_HOME", slicerHome))
      {
      vtkDebugMacro("Can't find SLICER_HOME env var. Can't source tcl scripts.");
      }
    else
      {
      // launch scripts

      std::string tclScript = slicerHome + "/../Slicer3/Modules/QueryAtlas/Tcl/QueryAtlas.tcl";
      app->Script ( "source %s", tclScript.c_str() );
      tclScript = slicerHome + "/../Slicer3/Modules/QueryAtlas/Tcl/QueryAtlasWeb.tcl";
      app->Script ( "source %s", tclScript.c_str() );
      tclScript = slicerHome + "/../Slicer3/Modules/QueryAtlas/Tcl/QueryAtlasControlledVocabulary.tcl";
      app->Script ( "source %s", tclScript.c_str() );
      tclScript = slicerHome + "/../Slicer3/Modules/QueryAtlas/Tcl/Card.tcl";
      app->Script ( "source %s", tclScript.c_str() );
      tclScript = slicerHome + "/../Slicer3/Modules/QueryAtlas/Tcl/CardFan.tcl";
      app->Script ( "source %s", tclScript.c_str() );

      // run init proc
      this->Script ( "QueryAtlasInit" );
      }
      */
      this->Script ( "source $env(SLICER_HOME)/../Slicer3/Modules/QueryAtlas/Tcl/QueryAtlas.tcl" );
      this->Script ( "QueryAtlasInit" );
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildAcknowledgementPanel ( )
{
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast(this->GetApplication());


    vtkKWLabel *abirn = vtkKWLabel::New();
    abirn->SetParent ( this->GetLogoFrame() );
    abirn->Create();
    abirn->SetImageToIcon ( this->GetAcknowledgementIcons()->GetBIRNLogo() );

    vtkKWLabel *anac = vtkKWLabel::New();
    anac->SetParent ( this->GetLogoFrame() );
    anac->Create();
    anac->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNACLogo() );

    vtkKWLabel *anamic = vtkKWLabel::New();
    anamic->SetParent ( this->GetLogoFrame() );
    anamic->Create();
    anamic->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNAMICLogo() );

    vtkKWLabel *aigt = vtkKWLabel::New();
    aigt->SetParent ( this->GetLogoFrame() );
    aigt->Create();
    aigt->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNCIGTLogo() );

    this->CollaboratorIcons = vtkQueryAtlasCollaboratorIcons::New();

    vtkKWLabel *abi = vtkKWLabel::New();
    abi->SetParent ( this->GetLogoFrame() );
    abi->Create();
    abi->SetImageToIcon ( this->GetCollaboratorIcons()->GetBrainInfoLogo() );
    
    app->Script ("grid %s -row 0 -column 0 -padx 2 -pady 2 -sticky w",  abirn->GetWidgetName());
    app->Script ("grid %s -row 0 -column 1 -padx 2 -pady 2 -sticky w", anamic->GetWidgetName());
    app->Script ("grid %s -row 0 -column 2 -padx 2 -pady 2 -sticky w",  anac->GetWidgetName());
    app->Script ("grid %s -row 1 -column 0 -padx 2 -pady 2 -sticky w",  aigt->GetWidgetName());                 
    app->Script ("grid %s -row 1 -column 1 -padx 2 -pady 2 -sticky w",  abi->GetWidgetName());

    abirn->Delete();
    anac->Delete();
    anamic->Delete();
    aigt->Delete();
    abi->Delete();
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildLoadAndConvertGUI ( )
{

  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "QueryAtlas" );

    //---
    // LOAD AND CONVERSION FRAME 
    //---
    vtkSlicerModuleCollapsibleFrame *convertFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    convertFrame->SetParent ( page );
    convertFrame->Create ( );
    convertFrame->SetLabelText ("Scene Setup");
    convertFrame->ExpandFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 4 -in %s",
                  convertFrame->GetWidgetName(),
                  this->UIPanel->GetPageWidget("QueryAtlas")->GetWidgetName());

    this->BuildLoaderContextButtons ( convertFrame->GetFrame() );
    
    vtkKWFrame *switcher = vtkKWFrame::New();
    switcher->SetParent ( convertFrame->GetFrame() );
    switcher->Create();
    
    this->BuildLoaderContextFrames ( switcher );
    this->BuildFreeSurferFIPSFrame ( );
    this->BuildQdecFrame ( );
    this->PackLoaderContextFrame ( this->FIPSFSFrame );
    app->Script ( "pack %s -side top -fill x -expand 1 -pady 0", switcher->GetWidgetName() );

    this->ColorCodeLoaderContextButtons ( this->FIPSFSButton );
    switcher->Delete();
    convertFrame->Delete();
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildFreeSurferFIPSFrame( )
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  
  vtkKWLabel *l = vtkKWLabel::New();
  l->SetParent ( this->FIPSFSFrame);
  l->Create();
  l->SetText("");
  this->Script ( "pack %s -side top -anchor c -padx 2", l->GetWidgetName() );
  l->Delete();  

    vtkSlicerPopUpHelpWidget *helpy = vtkSlicerPopUpHelpWidget::New();
    helpy->SetParent ( this->FIPSFSFrame );
    helpy->Create ( );
    helpy->SetHelpText (" This is some sample help text. This is some more sample help text. Not so very helpful yet." );
    helpy->SetHelpTitle ( "Testing Popup Help" );
    this->Script ( "pack %s -side top -anchor c -padx 2 -pady 6", helpy->GetWidgetName() );
    helpy->Delete();

    this->FSbrainSelector = vtkSlicerNodeSelectorWidget::New() ;
    this->FSbrainSelector->SetParent ( this->FIPSFSFrame );
    this->FSbrainSelector->Create ( );
    this->FSbrainSelector->SetNodeClass("vtkMRMLVolumeNode", NULL, NULL, NULL);
    this->FSbrainSelector->SetMRMLScene(this->GetMRMLScene());
    this->FSbrainSelector->SetBorderWidth(2);
    this->FSbrainSelector->SetPadX(2);
    this->FSbrainSelector->SetPadY(2);
    this->FSbrainSelector->GetWidget()->GetWidget()->IndicatorVisibilityOff();
    this->FSbrainSelector->GetWidget()->GetWidget()->SetWidth(20);
    this->FSbrainSelector->GetLabel()->SetWidth(18);
    this->FSbrainSelector->SetLabelText( "Anatomical volume: ");
    this->FSbrainSelector->SetBalloonHelpString("select a volume (FreeSurfer brain.mgz) from the current  scene.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                   this->FSbrainSelector->GetWidgetName());

    this->FSasegSelector = vtkSlicerNodeSelectorWidget::New() ;
    this->FSasegSelector->SetParent ( this->FIPSFSFrame );
    this->FSasegSelector->Create ( );
    this->FSasegSelector->SetNodeClass("vtkMRMLVolumeNode", "LabelMap", "1", NULL);
    this->FSasegSelector->SetMRMLScene(this->GetMRMLScene());
    this->FSasegSelector->SetBorderWidth(2);
    this->FSasegSelector->SetPadX(2);
    this->FSasegSelector->SetPadY(2);
    this->FSasegSelector->GetWidget()->GetWidget()->IndicatorVisibilityOff();
    this->FSasegSelector->GetWidget()->GetWidget()->SetWidth(20);
    this->FSasegSelector->GetLabel()->SetWidth(18);
    this->FSasegSelector->SetLabelText( "Annotated labelmap: ");
    this->FSasegSelector->SetBalloonHelpString("select an annotated label map (FreeSurfer aparc+aseg) from the current  scene.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                   this->FSasegSelector->GetWidgetName());

    this->FSmodelSelector = vtkSlicerNodeSelectorWidget::New() ;
    this->FSmodelSelector->SetParent ( this->FIPSFSFrame );
    this->FSmodelSelector->Create ( );
    this->FSmodelSelector->SetNodeClass("vtkMRMLModelNode", NULL, NULL, NULL);
    this->FSmodelSelector->SetMRMLScene(this->GetMRMLScene());
    this->FSmodelSelector->SetBorderWidth(2);
    this->FSmodelSelector->SetPadX(2);
    this->FSmodelSelector->SetPadY(2);
    this->FSmodelSelector->GetWidget()->GetWidget()->IndicatorVisibilityOff();
    this->FSmodelSelector->GetWidget()->GetWidget()->SetWidth(20);
    this->FSmodelSelector->GetLabel()->SetWidth(18);
    this->FSmodelSelector->SetLabelText( "Annotated model: ");
    this->FSmodelSelector->SetBalloonHelpString("select an annotated model (FreeSurfer lh.pial) from the current  scene.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->FSmodelSelector->GetWidgetName());

    this->FSstatsSelector = vtkSlicerNodeSelectorWidget::New() ;
    this->FSstatsSelector->SetParent ( this->FIPSFSFrame );
    this->FSstatsSelector->Create ( );
    this->FSstatsSelector->SetNodeClass("vtkMRMLVolumeNode", NULL, NULL, NULL);
    this->FSstatsSelector->SetMRMLScene(this->GetMRMLScene());
    this->FSstatsSelector->SetBorderWidth(2);
    this->FSstatsSelector->SetPadX(2);
    this->FSstatsSelector->SetPadY(2);
    this->FSstatsSelector->GetWidget()->GetWidget()->IndicatorVisibilityOff();
    this->FSstatsSelector->GetWidget()->GetWidget()->SetWidth(20);
    this->FSstatsSelector->GetLabel()->SetWidth(18);
    this->FSstatsSelector->SetLabelText( "Statistics: ");
    this->FSstatsSelector->SetBalloonHelpString("select a statistical overlay volume from the current  scene.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->FSstatsSelector->GetWidgetName());

    this->FSgoButton = vtkKWPushButton::New();
    this->FSgoButton->SetParent ( this->FIPSFSFrame );
    this->FSgoButton->Create();
    this->FSgoButton->SetText ("Set up query scene");
    this->FSgoButton->SetWidth ( 20 );
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->FSgoButton->GetWidgetName());
    

    this->ProcessGUIEvents ( this->FSbrainSelector, vtkSlicerNodeSelectorWidget::NodeSelectedEvent, NULL );
    this->ProcessGUIEvents ( this->FSmodelSelector, vtkSlicerNodeSelectorWidget::NodeSelectedEvent, NULL );
    this->ProcessGUIEvents ( this->FSasegSelector, vtkSlicerNodeSelectorWidget::NodeSelectedEvent, NULL );
    this->ProcessGUIEvents ( this->FSstatsSelector, vtkSlicerNodeSelectorWidget::NodeSelectedEvent, NULL );
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildQdecFrame ( )
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  
  vtkKWLabel *l = vtkKWLabel::New();
  l->SetParent ( this->QdecFrame);
  l->Create();
  l->SetText("");
  this->Script ( "pack %s -side top -anchor c -padx 2", l->GetWidgetName() );
  l->Delete();  

    vtkSlicerPopUpHelpWidget *helpy = vtkSlicerPopUpHelpWidget::New();
    helpy->SetParent ( this->QdecFrame );
    helpy->Create ( );
    helpy->SetHelpText (" This is some sample help text. This is some more sample help text. Not so very helpful yet." );
    helpy->SetHelpTitle ( "Testing Popup Help" );
    this->Script ( "pack %s -side top -anchor c -padx 2 -pady 6", helpy->GetWidgetName() );
    helpy->Delete();

    this->QdecGetResultsButton = vtkKWLoadSaveButtonWithLabel::New() ;
    this->QdecGetResultsButton->SetParent ( this->QdecFrame );
    this->QdecGetResultsButton->Create();
    this->QdecGetResultsButton->GetWidget()->SetImageToIcon ( app->GetApplicationGUI()->GetApplicationToolbar()->GetSlicerToolbarIcons()->GetLoadSceneIcon() );   
    this->QdecGetResultsButton->GetWidget()->SetBorderWidth(0);
    this->QdecGetResultsButton->GetWidget()->SetReliefToFlat ( );
    this->QdecGetResultsButton->SetBalloonHelpString ( "Load Qdec results" );
    //this->QdecGetResultsButton->GetWidget()->SetCommand ( this, "" );
    this->QdecGetResultsButton->GetWidget()->GetLoadSaveDialog()->SetTitle("Load Qdec results");
    this->QdecGetResultsButton->GetLabel()->SetText( "Qdec results: ");
    this->QdecGetResultsButton->GetLabel()->SetWidth ( 14 );
    this->QdecGetResultsButton->GetWidget()->GetLoadSaveDialog()->ChooseDirectoryOn();
    this->QdecGetResultsButton->GetWidget()->GetLoadSaveDialog()->SaveDialogOff();
    //this->QdecGetResultsButton->GetLoadSaveDialog()->SetFileTypes ( "");
    this->QdecGetResultsButton->SetBalloonHelpString("Load all results from previous Qdec analysis if not already present in the scene.");
    this->Script ( "pack %s -side top -anchor nw -padx 6 -pady 2",
                  this->QdecGetResultsButton->GetWidgetName());

    this->QdecModelSelector = vtkSlicerNodeSelectorWidget::New() ;
    this->QdecModelSelector->SetParent ( this->QdecFrame );
    this->QdecModelSelector->Create ( );
    this->QdecModelSelector->SetNodeClass("vtkMRMLModelNode", NULL, NULL, NULL);
    this->QdecModelSelector->SetMRMLScene(this->GetMRMLScene());
    this->QdecModelSelector->SetBorderWidth(2);
    this->QdecModelSelector->SetPadX(2);
    this->QdecModelSelector->SetPadY(2);
    this->QdecModelSelector->GetWidget()->GetWidget()->IndicatorVisibilityOff();
    this->QdecModelSelector->GetWidget()->GetWidget()->SetWidth(20);
    this->QdecModelSelector->GetLabel()->SetWidth(14);
    this->QdecModelSelector->SetLabelText( "Annotated model: ");
    this->QdecModelSelector->SetBalloonHelpString("select an annotated model (FreeSurfer inflated.pial) from the current  scene.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 4",
                  this->QdecModelSelector->GetWidgetName());


    this->QdecScalarSelector = vtkKWMenuButtonWithLabel::New();
    this->QdecScalarSelector->SetParent ( this->QdecFrame );
    this->QdecScalarSelector->Create ( );
    this->QdecScalarSelector->SetBorderWidth(2);
    this->QdecScalarSelector->SetPadX(2);
    this->QdecScalarSelector->SetPadY(2);
    this->QdecScalarSelector->GetWidget()->SetWidth(20);
    this->QdecScalarSelector->GetLabel()->SetWidth(14);
    this->QdecScalarSelector->GetLabel()->SetText( "Select overlay: ");
    this->QdecScalarSelector->SetBalloonHelpString("select a scalar overlay for this model.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->QdecScalarSelector->GetWidgetName());

    this->QdecGoButton = vtkKWPushButton::New();
    this->QdecGoButton->SetParent ( this->QdecFrame );
    this->QdecGoButton->Create();
    this->QdecGoButton->SetText ("Set up query scene");
    this->QdecGoButton->SetWidth ( 20 );
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                   this->QdecGoButton->GetWidgetName());



    //get uri from model, load annotations from a relative path??

}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildAnnotationOptionsGUI ( )
{
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "QueryAtlas" );
    // -------------------------------------------------------------------------------------------------
    // ---
    // ANNOTATION OPTIONS FRAME

    // ---
    // -------------------------------------------------------------------------------------------------
    vtkSlicerModuleCollapsibleFrame *annotationFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    annotationFrame->SetParent ( page );
    annotationFrame->Create ();
    annotationFrame->SetLabelText ( "Annotation & Display Options" );
    annotationFrame->CollapseFrame ( );

    vtkKWLabel *annoLabel = vtkKWLabel::New();
    annoLabel->SetParent ( annotationFrame->GetFrame() );
    annoLabel->Create();
    annoLabel->SetText ("annotation visibility: " );

    this->AnnotationVisibilityButton = vtkKWPushButton::New();
    this->AnnotationVisibilityButton->SetParent ( annotationFrame->GetFrame() );
    this->AnnotationVisibilityButton->Create();
    // get the icon this way; don't seem to admit baseGUI scope.
    // TODO: move common icons up into applicationGUI for easy access.
    vtkKWIcon *i = app->GetApplicationGUI()->GetMainSliceGUI0()->GetSliceController()->GetVisibilityIcons()->GetVisibleIcon();
    this->AnnotationVisibilityButton->SetImageToIcon ( i );
    this->AnnotationVisibilityButton->SetBorderWidth ( 0 );
    this->AnnotationVisibilityButton->SetReliefToFlat();    
    this->AnnotationVisibilityButton->SetBalloonHelpString ( "Toggle annotation visibility." );

    vtkKWLabel *modelLabel = vtkKWLabel::New();
    modelLabel->SetParent ( annotationFrame->GetFrame() );
    modelLabel->Create();
    modelLabel->SetText ("model visibility: " );

    this->ModelVisibilityButton = vtkKWPushButton::New();
    this->ModelVisibilityButton->SetParent ( annotationFrame->GetFrame() );
    this->ModelVisibilityButton->Create();
    // get the icon this way; don't seem to admit baseGUI scope.
    // TODO: move common icons up into applicationGUI for easy access.
    i = app->GetApplicationGUI()->GetMainSliceGUI0()->GetSliceController()->GetVisibilityIcons()->GetVisibleIcon();
    this->ModelVisibilityButton->SetImageToIcon ( i );
    this->ModelVisibilityButton->SetBorderWidth ( 0 );
    this->ModelVisibilityButton->SetReliefToFlat();    
    this->ModelVisibilityButton->SetBalloonHelpString ( "Toggle model visibility." );

    vtkKWLabel *l = vtkKWLabel::New();
    l->SetParent ( annotationFrame->GetFrame() );
    l->Create ( );
    l->SetText ( "annotation term set: " );

    this->AnnotationTermSetMenuButton = vtkKWMenuButton::New();
    this->AnnotationTermSetMenuButton->SetParent ( annotationFrame->GetFrame() );
    this->AnnotationTermSetMenuButton->Create();
    this->AnnotationTermSetMenuButton->SetWidth ( 25 );
    this->AnnotationTermSetMenuButton->GetMenu()->AddRadioButton ("local identifier");
    this->AnnotationTermSetMenuButton->GetMenu()->AddRadioButton ("BIRNLex String");
    this->AnnotationTermSetMenuButton->GetMenu()->AddRadioButton ("NeuroNames String");
    this->AnnotationTermSetMenuButton->GetMenu()->AddRadioButton ("UMLS CID");
    this->AnnotationTermSetMenuButton->GetMenu()->AddSeparator();
    this->AnnotationTermSetMenuButton->GetMenu()->AddCommand ( "close" );    
    this->AnnotationTermSetMenuButton->GetMenu()->SelectItem ("local identifier");

    app->Script ( "grid %s -row 0 -column 0 -sticky nse -padx 2 -pady 2",
                  l->GetWidgetName() );
    app->Script ( "grid %s -row 0 -column 1 -sticky wns -padx 2 -pady 2",
                  this->AnnotationTermSetMenuButton->GetWidgetName() );
    app->Script ( "grid %s -row 1 -column 0 -sticky ens -padx 2 -pady 2",
                  annoLabel->GetWidgetName() );
    app->Script ( "grid %s -row 1 -column 1  -sticky wns -padx 2 -pady 2",
                  this->AnnotationVisibilityButton->GetWidgetName() );
    app->Script ( "grid %s -row 2 -column 0   -sticky ens -padx 2 -pady 2",
                  modelLabel->GetWidgetName() );
    app->Script ( "grid %s -row 2 -column 1   -sticky wns -padx 2 -pady 2",
                  this->ModelVisibilityButton->GetWidgetName() );

    app->Script ( "pack %s -side top -anchor nw -fill x -expand y -padx 4 -pady 2 -in %s",
                  annotationFrame->GetWidgetName(), 
                  this->UIPanel->GetPageWidget("QueryAtlas")->GetWidgetName());

    l->Delete();
    annoLabel->Delete();
    modelLabel->Delete();
    annotationFrame->Delete();
}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildOntologyGUI ( )
{
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "QueryAtlas" );
    // -------------------------------------------------------------------------------------------------
    // ---
    // ONTOLOGY SEARCH FRAME
    // ---
    // -------------------------------------------------------------------------------------------------
    vtkSlicerModuleCollapsibleFrame *hierarchyFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    hierarchyFrame->SetParent ( page );
    hierarchyFrame->Create ( );
    hierarchyFrame->SetLabelText ("Ontology Mapping");
    hierarchyFrame->CollapseFrame ( );

    // first row (local terms)
    vtkKWLabel *localLabel = vtkKWLabel::New();
    localLabel->SetParent ( hierarchyFrame->GetFrame() );
    localLabel->Create();
    localLabel->SetText ("local term: ");
    this->LocalSearchTermEntry = vtkKWEntry::New();
    this->LocalSearchTermEntry->SetParent ( hierarchyFrame->GetFrame() );
    this->LocalSearchTermEntry->Create();
    this->LocalSearchTermEntry->SetValue ("");
    this->AddLocalTermButton = vtkKWPushButton::New();
    this->AddLocalTermButton->SetParent ( hierarchyFrame->GetFrame() );
    this->AddLocalTermButton->Create();
    this->AddLocalTermButton->SetImageToIcon ( this->QueryAtlasIcons->GetAddIcon() );
    this->AddLocalTermButton->SetBorderWidth ( 0 );
    this->AddLocalTermButton->SetReliefToFlat();
    this->AddLocalTermButton->SetBalloonHelpString ("Save this term for building queries.");
    
    // second row (synonyms)
    vtkKWLabel *synLabel = vtkKWLabel::New();
    synLabel->SetParent ( hierarchyFrame->GetFrame() );
    synLabel->Create();
    synLabel->SetText ("synonyms: ");
    this->SynonymsMenuButton = vtkKWMenuButton::New();
    this->SynonymsMenuButton->SetParent ( hierarchyFrame->GetFrame() );
    this->SynonymsMenuButton->Create();
    this->SynonymsMenuButton->IndicatorVisibilityOn();
    this->AddSynonymButton = vtkKWPushButton::New();
    this->AddSynonymButton->SetParent ( hierarchyFrame->GetFrame() );
    this->AddSynonymButton->Create();
    this->AddSynonymButton->SetImageToIcon ( this->QueryAtlasIcons->GetAddIcon() );
    this->AddSynonymButton->SetBorderWidth ( 0 );
    this->AddSynonymButton->SetReliefToFlat();
    this->AddSynonymButton->SetBalloonHelpString ("Save this term for building queries.");

    // third row (BIRNLex)
    vtkKWLabel *birnLabel = vtkKWLabel::New();
    birnLabel->SetParent ( hierarchyFrame->GetFrame() );
    birnLabel->Create();
    birnLabel->SetText ("BIRNLex: ");
    this->BIRNLexEntry = vtkKWEntry::New();
    this->BIRNLexEntry->SetParent ( hierarchyFrame->GetFrame() );
    this->BIRNLexEntry->Create();
    this->BIRNLexEntry->SetValue ("");
    this->AddBIRNLexStringButton = vtkKWPushButton::New();
    this->AddBIRNLexStringButton->SetParent ( hierarchyFrame->GetFrame() );
    this->AddBIRNLexStringButton->Create();
    this->AddBIRNLexStringButton->SetImageToIcon ( this->QueryAtlasIcons->GetAddIcon() );
    this->AddBIRNLexStringButton->SetBorderWidth(0);
    this->AddBIRNLexStringButton->SetReliefToFlat();
    this->AddBIRNLexStringButton->SetBalloonHelpString ("Save this term for building queries.");
    this->BIRNLexHierarchyButton = vtkKWPushButton::New();
    this->BIRNLexHierarchyButton->SetParent ( hierarchyFrame->GetFrame() );
    this->BIRNLexHierarchyButton->Create();
    this->BIRNLexHierarchyButton->SetImageToIcon ( this->QueryAtlasIcons->GetOntologyBrowserIcon() );
    this->BIRNLexHierarchyButton->SetBorderWidth ( 0 );
    this->BIRNLexHierarchyButton->SetReliefToFlat ( );
    this->BIRNLexHierarchyButton->SetBalloonHelpString ("View in BIRNLex ontology browser.");

    // forthrow (BIRNLex ID )
    vtkKWLabel *birnidLabel = vtkKWLabel::New();
    birnidLabel->SetParent ( hierarchyFrame->GetFrame() );
    birnidLabel->Create();
    birnidLabel->SetText ("BIRNLex ID: ");
    this->BIRNLexIDEntry = vtkKWEntry::New();
    this->BIRNLexIDEntry->SetParent ( hierarchyFrame->GetFrame() );
    this->BIRNLexIDEntry->Create();
    this->BIRNLexIDEntry->SetValue("");
    this->AddBIRNLexIDButton = vtkKWPushButton::New();
    this->AddBIRNLexIDButton->SetParent ( hierarchyFrame->GetFrame() );
    this->AddBIRNLexIDButton->Create();
    this->AddBIRNLexIDButton->SetImageToIcon ( this->QueryAtlasIcons->GetAddIcon() );
    this->AddBIRNLexIDButton->SetBorderWidth ( 0 );
    this->AddBIRNLexIDButton->SetReliefToFlat ( );
    this->AddBIRNLexIDButton->SetBalloonHelpString ("Save this term for building queries.");

    // fifth row (NeuroNames)
    vtkKWLabel *nnLabel = vtkKWLabel::New();
    nnLabel->SetParent ( hierarchyFrame->GetFrame() );
    nnLabel->Create();
    nnLabel->SetText ("NeuroNames: ");
    this->NeuroNamesEntry = vtkKWEntry::New();
    this->NeuroNamesEntry->SetParent ( hierarchyFrame->GetFrame() );
    this->NeuroNamesEntry->Create();
    this->NeuroNamesEntry->SetValue ("");
    this->AddNeuroNamesStringButton = vtkKWPushButton::New();
    this->AddNeuroNamesStringButton->SetParent ( hierarchyFrame->GetFrame() );
    this->AddNeuroNamesStringButton->Create();
    this->AddNeuroNamesStringButton->SetImageToIcon ( this->QueryAtlasIcons->GetAddIcon() );
    this->AddNeuroNamesStringButton->SetBorderWidth (0 );
    this->AddNeuroNamesStringButton->SetReliefToFlat( );
    this->AddNeuroNamesStringButton->SetBalloonHelpString ("Save this term for building queries.");
    this->NeuroNamesHierarchyButton = vtkKWPushButton::New();
    this->NeuroNamesHierarchyButton->SetParent ( hierarchyFrame->GetFrame() );
    this->NeuroNamesHierarchyButton->Create();
    this->NeuroNamesHierarchyButton->SetImageToIcon ( this->QueryAtlasIcons->GetOntologyBrowserIcon() );
    this->NeuroNamesHierarchyButton->SetBorderWidth ( 0 );
    this->NeuroNamesHierarchyButton->SetReliefToFlat();
    this->NeuroNamesHierarchyButton->SetBalloonHelpString ("View in NeuroNames ontology browser.");

    // sixth row (NeuroNames ID)
    vtkKWLabel *nnidLabel = vtkKWLabel::New();    
    nnidLabel->SetParent ( hierarchyFrame->GetFrame() );
    nnidLabel->Create();
    nnidLabel->SetText ("NeuroNames ID: ");
    this->NeuroNamesIDEntry = vtkKWEntry::New();
    this->NeuroNamesIDEntry->SetParent ( hierarchyFrame->GetFrame() );
    this->NeuroNamesIDEntry->Create();
    this->NeuroNamesIDEntry->SetValue ("");
    this->AddNeuroNamesIDButton = vtkKWPushButton::New();
    this->AddNeuroNamesIDButton->SetParent ( hierarchyFrame->GetFrame() );
    this->AddNeuroNamesIDButton->Create();
    this->AddNeuroNamesIDButton->SetImageToIcon ( this->QueryAtlasIcons->GetAddIcon() );
    this->AddNeuroNamesIDButton->SetBorderWidth(0);
    this->AddNeuroNamesIDButton->SetReliefToFlat();
    this->AddNeuroNamesIDButton->SetBalloonHelpString ("Save this term for building queries.");
    // seventh row (UMLS)
    vtkKWLabel *umlsLabel = vtkKWLabel::New();
    umlsLabel->SetParent ( hierarchyFrame->GetFrame() );
    umlsLabel->Create();
    umlsLabel->SetText ("UMLS CID: ");
    this->UMLSCIDEntry = vtkKWEntry::New();
    this->UMLSCIDEntry->SetParent ( hierarchyFrame->GetFrame() );
    this->UMLSCIDEntry->Create();
    this->UMLSCIDEntry->SetValue ("");
    this->AddUMLSCIDButton = vtkKWPushButton::New();
    this->AddUMLSCIDButton->SetParent ( hierarchyFrame->GetFrame() );
    this->AddUMLSCIDButton->Create();
    this->AddUMLSCIDButton->SetImageToIcon ( this->QueryAtlasIcons->GetAddIcon() );
    this->AddUMLSCIDButton->SetBorderWidth(0);
    this->AddUMLSCIDButton->SetReliefToFlat();
    this->AddUMLSCIDButton->SetBalloonHelpString ("Save this term for building queries.");
    this->UMLSHierarchyButton = vtkKWPushButton::New();
    this->UMLSHierarchyButton->SetParent ( hierarchyFrame->GetFrame() );
    this->UMLSHierarchyButton->Create();
    this->UMLSHierarchyButton->SetImageToIcon ( this->QueryAtlasIcons->GetOntologyBrowserDisabledIcon() );
    this->UMLSHierarchyButton->SetBorderWidth ( 0 );
    this->UMLSHierarchyButton->SetReliefToFlat();
    this->UMLSHierarchyButton->SetBalloonHelpString ("View in UMLS ontology browser.");
    
    // eighth row (listbox saved terms)
    vtkKWLabel *termsLabel = vtkKWLabel::New();
    termsLabel->SetParent (hierarchyFrame->GetFrame() );
    termsLabel->Create();
    termsLabel->SetText ( "(structure) ");
    
    vtkKWFrame *f = vtkKWFrame::New();
    f->SetParent ( hierarchyFrame->GetFrame() );
    f->Create();
    
    this->SavedTerms = vtkQueryAtlasSearchTermWidget::New();
    this->SavedTerms->SetParent ( f );
    this->SavedTerms->Create();
//    int i = this->SavedTerms->GetMultiColumnList()->GetWidget()->GetColumnIndexWithName ( "Search terms" );
//    this->SavedTerms->GetMultiColumnList()->GetWidget()->SetColumnName ( i, "Saved terms");
    this->SavedTerms->GetMultiColumnList()->GetWidget()->SetHeight(3);
    
    //---
    // grid and pack up
    //---
    app->Script ( "grid columnconfigure %s 0 -weight 0", hierarchyFrame->GetFrame()->GetWidgetName() );
    app->Script ( "grid columnconfigure %s 1 -weight 1", hierarchyFrame->GetFrame()->GetWidgetName() );
    app->Script ( "grid columnconfigure %s 2 -weight 0", hierarchyFrame->GetFrame()->GetWidgetName() );
    app->Script ( "grid columnconfigure %s 3 -weight 0", hierarchyFrame->GetFrame()->GetWidgetName() );

    app->Script ( "grid %s -row 0 -column 0 -sticky e -padx 0 -pady 1", localLabel->GetWidgetName() );
    app->Script ( "grid %s -row 1 -column 0 -sticky e -padx 0 -pady 1", synLabel->GetWidgetName() );
    app->Script ( "grid %s -row 2 -column 0 -sticky e -padx 0 -pady 1", birnLabel->GetWidgetName() );
    app->Script ( "grid %s -row 3 -column 0 -sticky e -padx 0 -pady 1", birnidLabel->GetWidgetName() );
    app->Script ( "grid %s -row 4 -column 0 -sticky e -padx 0 -pady 1", nnLabel->GetWidgetName() );
    app->Script ( "grid %s -row 5 -column 0 -sticky e -padx 0 -pady 1", nnidLabel->GetWidgetName() );
    app->Script ( "grid %s -row 6 -column 0 -sticky e -padx 0 -pady 1", umlsLabel->GetWidgetName() );
    app->Script ( "grid %s -row 7 -column 0 -sticky ne -padx 0 -pady 1", termsLabel->GetWidgetName() );

    app->Script ( "grid %s -row 0 -column 1 -sticky ew -padx 2 -pady 1", this->LocalSearchTermEntry->GetWidgetName() );
    app->Script ( "grid %s -row 1 -column 1 -sticky ew -padx 2 -pady 1", this->SynonymsMenuButton->GetWidgetName() );
    app->Script ( "grid %s -row 2 -column 1 -sticky ew -padx 2 -pady 1", this->BIRNLexEntry->GetWidgetName() );
    app->Script ( "grid %s -row 3 -column 1 -sticky ew -padx 2 -pady 1", this->BIRNLexIDEntry->GetWidgetName() );
    app->Script ( "grid %s -row 4 -column 1 -sticky ew -padx 2 -pady 1", this->NeuroNamesEntry->GetWidgetName() );
    app->Script ( "grid %s -row 5 -column 1 -sticky ew -padx 2 -pady 1", this->NeuroNamesIDEntry->GetWidgetName() );
    app->Script ( "grid %s -row 6 -column 1 -sticky ew -padx 2 -pady 1", this->UMLSCIDEntry->GetWidgetName() );
    app->Script ( "grid %s -row 7 -column 1 -sticky ew -columnspan 2 -padx 2 -pady 1", f->GetWidgetName() );
    app->Script ( "pack %s -side top -fill x -expand true -padx 0 -pady 0", this->SavedTerms->GetWidgetName() );
    f->Delete();
    
    app->Script ( "grid %s -row 0 -column 2 -padx 2 -pady 1", this->AddLocalTermButton->GetWidgetName() );
    app->Script ( "grid %s -row 1 -column 2 -padx 2 -pady 1", this->AddSynonymButton->GetWidgetName() );
    app->Script ( "grid %s -row 2 -column 2 -padx 2 -pady 1", this->AddBIRNLexStringButton->GetWidgetName() );
    app->Script ( "grid %s -row 3 -column 2 -padx 2 -pady 1", this->AddBIRNLexIDButton->GetWidgetName() );
    app->Script ( "grid %s -row 4 -column 2 -padx 2 -pady 1", this->AddNeuroNamesStringButton->GetWidgetName() );
    app->Script ( "grid %s -row 5 -column 2 -padx 2 -pady 1", this->AddNeuroNamesIDButton->GetWidgetName() );
    app->Script ( "grid %s -row 6 -column 2 -padx 2 -pady 1", this->AddUMLSCIDButton->GetWidgetName() );

    app->Script ( "grid %s -row 2 -column 3 -padx 2 -pady 1", this->BIRNLexHierarchyButton->GetWidgetName() );
    app->Script ( "grid %s -row 4 -column 3 -padx 2 -pady 1", this->NeuroNamesHierarchyButton->GetWidgetName() );
    app->Script ( "grid %s -row 6 -column 3 -padx 2 -pady 1", this->UMLSHierarchyButton->GetWidgetName() );

    app->Script ( "pack %s -side top -anchor nw -fill x -expand y -padx 2 -pady 2 -in %s",
                  hierarchyFrame->GetWidgetName(), 
                  this->UIPanel->GetPageWidget("QueryAtlas")->GetWidgetName());


    //---
    // clean up.
    //---
    localLabel->Delete();
    synLabel->Delete();
    birnLabel->Delete();
    birnidLabel->Delete();
    nnLabel->Delete();
    nnidLabel->Delete();
    umlsLabel->Delete();
    termsLabel->Delete();
    hierarchyFrame->Delete();
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildSearchTermGUI ( )
{
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "QueryAtlas" );
    // -------------------------------------------------------------------------------------------------
    // ---
    // BUILD SEARCH TERMS FRAME
    // ---
    // -------------------------------------------------------------------------------------------------
    vtkSlicerModuleCollapsibleFrame *queryFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    queryFrame->SetParent ( page );
    queryFrame->Create ( );
    queryFrame->SetLabelText ("Search Terms");
    queryFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  queryFrame->GetWidgetName(),
                  this->UIPanel->GetPageWidget("QueryAtlas")->GetWidgetName());

    this->BuildQueryBuilderContextButtons ( queryFrame->GetFrame() );
    

    this->SwitchQueryFrame = vtkKWFrame::New();
    this->SwitchQueryFrame->SetParent ( queryFrame->GetFrame() );
    this->SwitchQueryFrame->Create();

    //--- these are the frames that populate the shared frame;
    //--- they are packed/unpacked depending on which context button is pushed.
    this->BuildQueryBuilderContextFrames ( this->SwitchQueryFrame );
    this->BuildSpeciesFrame();
    this->BuildPopulationFrame();
    this->BuildStructureFrame();
    this->BuildOtherFrame();
    this->PackQueryBuilderContextFrame ( this->StructureFrame );
    app->Script ( "pack %s -side top -fill x -expand 1", this->SwitchQueryFrame->GetWidgetName() );
//    this->Script ( "place %s -relx 0 -rely 0 -anchor nw", this->SwitchQueryFrame->GetWidgetName());
    this->ColorCodeContextButtons ( this->StructureButton );
    queryFrame->Delete();

}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildQueriesGUI ( )
{
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "QueryAtlas" );

    // -------------------------------------------------------------------------------------------------
    // ---
    // BUILD QUERIES FRAME
    // ---
    // -------------------------------------------------------------------------------------------------
    vtkSlicerModuleCollapsibleFrame *searchFrame = vtkSlicerModuleCollapsibleFrame::New();
    searchFrame->SetParent ( page);
    searchFrame->Create();
    searchFrame->SetLabelText ("Build Queries");
    searchFrame->CollapseFrame();
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  searchFrame->GetWidgetName(),
                  this->UIPanel->GetPageWidget("QueryAtlas")->GetWidgetName());

    /*
    vtkKWFrame *f = vtkKWFrame::New();
    f->SetParent ( searchFrame->GetFrame() );
    f->Create();
    this->ResultsWithAnyButton = vtkKWRadioButton::New();
    this->ResultsWithAnyButton->SetParent ( f );
    this->ResultsWithAnyButton->Create();
    this->ResultsWithAnyButton->SetImageToIcon ( this->QueryAtlasIcons->GetWithAnyIcon() );
    this->ResultsWithAnyButton->SetSelectImageToIcon ( this->QueryAtlasIcons->GetWithAnyDisabledIcon() );
    this->ResultsWithAnyButton->SetBorderWidth ( 0 );
    this->ResultsWithAnyButton->SetReliefToFlat ( );
    this->ResultsWithAnyButton->SetSelectedState ( 1 );
    this->ResultsWithAnyButton->IndicatorVisibilityOff();
    this->ResultsWithAnyButton->SetValueAsInt ( vtkQueryAtlasGUI::Or );
    this->ResultsWithAnyButton->SetBalloonHelpString ( "Search for results that include any of the search terms." );

    this->ResultsWithAllButton = vtkKWRadioButton::New();
    this->ResultsWithAllButton->SetParent ( f );
    this->ResultsWithAllButton->Create();
    this->ResultsWithAllButton->SetImageToIcon ( this->QueryAtlasIcons->GetWithAllIcon() );
    this->ResultsWithAllButton->SetImageToIcon ( this->QueryAtlasIcons->GetWithAllDisabledIcon() );
    this->ResultsWithAllButton->SetBorderWidth ( 0 );
    this->ResultsWithAllButton->SetReliefToFlat ( );
    this->ResultsWithAllButton->SetStateToDisabled();
    this->ResultsWithAllButton->SetValueAsInt ( vtkQueryAtlasGUI::And );
    this->ResultsWithAllButton->SetVariableName ( this->ResultsWithAnyButton->GetVariableName() );
    this->ResultsWithAllButton->SetBalloonHelpString ( "Search for results that include all of the search terms (disabled)." );

    this->ResultsWithExactButton = vtkKWRadioButton::New();
    this->ResultsWithExactButton->SetParent ( f );
    this->ResultsWithExactButton->Create();
    this->ResultsWithExactButton->SetImageToIcon ( this->QueryAtlasIcons->GetWithExactIcon() );
    this->ResultsWithExactButton->SetImageToIcon ( this->QueryAtlasIcons->GetWithExactDisabledIcon() );
    this->ResultsWithExactButton->SetBorderWidth ( 0 );
    this->ResultsWithExactButton->SetReliefToFlat ( );
    this->ResultsWithExactButton->SetStateToDisabled();
    this->ResultsWithExactButton->SetValueAsInt ( vtkQueryAtlasGUI::Quote );
    this->ResultsWithExactButton->SetVariableName ( this->ResultsWithAnyButton->GetVariableName() );
    this->ResultsWithExactButton->SetBalloonHelpString ( "Search for results that include the exact search terms (disabled)." );
    */

    vtkKWLabel *sl = vtkKWLabel::New();
    sl->SetParent ( searchFrame->GetFrame() );
    sl->Create();
    sl->SetText ("search target: ");
    this->DatabasesMenuButton = vtkKWMenuButton::New();
    this->DatabasesMenuButton->SetParent ( searchFrame->GetFrame() );
    this->DatabasesMenuButton->Create();
    this->DatabasesMenuButton->SetWidth (24);    
    this->BuildDatabasesMenu(this->DatabasesMenuButton->GetMenu() );
    this->SearchButton = vtkKWPushButton::New();
//    this->SearchButton->SetParent ( f );
    this->SearchButton->SetParent ( searchFrame->GetFrame() );
    this->SearchButton->Create();
    this->SearchButton->SetImageToIcon ( this->QueryAtlasIcons->GetSearchIcon() );
    this->SearchButton->SetBorderWidth ( 0 );
    this->SearchButton->SetReliefToFlat();
    this->SearchButton->SetBalloonHelpString ( "Perform a search" );
/*
    app->Script ( "pack %s %s %s %s -side left -anchor w -padx 2 -pady 2",
                  this->ResultsWithAnyButton->GetWidgetName(),
                  this->ResultsWithAllButton->GetWidgetName(),
                  this->ResultsWithExactButton->GetWidgetName(),
                  this->SearchButton->GetWidgetName() );
*/
    app->Script ("grid %s -row 0 -column 0 -padx 0 -pady 2 -sticky w",
                 sl->GetWidgetName() );
    app->Script ("grid %s -row 0 -column 1 -padx 0 -pady 2 -sticky w",
                 this->DatabasesMenuButton->GetWidgetName() );    
    app->Script ("grid %s -row 0 -column 2 -padx 2 -pady 2 -sticky w",
                 this->SearchButton->GetWidgetName() );
/*
    app->Script ("grid %s -row 1 -column 1 -padx 0 -pady 2 -sticky w",
                 f->GetWidgetName() );
*/


    // ---
    // QUERY RESULTS MANAGER FRAME
    // ---

    vtkKWFrame *managerFrame = vtkKWFrame::New();
    managerFrame->SetParent ( searchFrame->GetFrame() );
    managerFrame->Create();

    vtkKWFrame *curF = vtkKWFrame::New();
    curF->SetParent ( managerFrame );
    curF->Create();
    vtkKWFrame *topcurF = vtkKWFrame::New();
    topcurF->SetParent ( managerFrame );
    topcurF->Create();
    vtkKWLabel *curL = vtkKWLabel::New();
    curL->SetParent ( topcurF );
    curL->Create();
    curL->SetWidth ( 45 );
    curL->SetText ( "Latest search results" );
    curL->SetBackgroundColor ( _br, _bg, _bb);

    this->CurrentResultsList = vtkKWListBoxWithScrollbars::New();
    this->CurrentResultsList->SetParent ( topcurF );
    this->CurrentResultsList->Create();
    this->CurrentResultsList->GetWidget()->SetSelectionModeToMultiple();
    this->CurrentResultsList->GetWidget()->SetWidth ( 45 );
    this->CurrentResultsList->GetWidget()->SetHeight (4 );
    this->CurrentResultsList->HorizontalScrollbarVisibilityOn();
    this->CurrentResultsList->VerticalScrollbarVisibilityOn();
    this->CurrentResultsList->GetWidget()->SetDoubleClickCommand (this, "OpenLinkFromCurrentList" );

    this->DeselectAllCurrentResultsButton = vtkKWPushButton::New();
    this->DeselectAllCurrentResultsButton->SetParent (curF);
    this->DeselectAllCurrentResultsButton->Create();
    this->DeselectAllCurrentResultsButton->SetImageToIcon ( this->QueryAtlasIcons->GetDeselectAllIcon() );
    this->DeselectAllCurrentResultsButton->SetBorderWidth ( 0 );
    this->DeselectAllCurrentResultsButton->SetReliefToFlat();    
    this->DeselectAllCurrentResultsButton->SetBalloonHelpString ( "Deselect all results");

    this->DeleteCurrentResultButton = vtkKWPushButton::New();
    this->DeleteCurrentResultButton->SetParent (curF);
    this->DeleteCurrentResultButton->Create();
    this->DeleteCurrentResultButton->SetImageToIcon ( this->QueryAtlasIcons->GetClearSelectedIcon() );
    this->DeleteCurrentResultButton->SetBorderWidth ( 0 );
    this->DeleteCurrentResultButton->SetReliefToFlat();    
    this->DeleteCurrentResultButton->SetBalloonHelpString ( "Delete selected results");

    this->DeleteAllCurrentResultsButton = vtkKWPushButton::New();
    this->DeleteAllCurrentResultsButton->SetParent (curF);
    this->DeleteAllCurrentResultsButton->Create();
    this->DeleteAllCurrentResultsButton->SetImageToIcon ( this->QueryAtlasIcons->GetClearAllIcon() );
    this->DeleteAllCurrentResultsButton->SetBorderWidth ( 0 );
    this->DeleteAllCurrentResultsButton->SetReliefToFlat();    
    this->DeleteAllCurrentResultsButton->SetBalloonHelpString ("Delete all results ");

    this->SaveCurrentResultsButton = vtkKWPushButton::New();
    this->SaveCurrentResultsButton->SetParent (curF);
    this->SaveCurrentResultsButton->Create();
    this->SaveCurrentResultsButton->SetImageToIcon ( this->QueryAtlasIcons->GetReserveURIsIcon() );
    this->SaveCurrentResultsButton->SetBorderWidth ( 0 );
    this->SaveCurrentResultsButton->SetReliefToFlat();    
    this->SaveCurrentResultsButton->SetBalloonHelpString ("Reserve all results");

    this->SaveCurrentSelectedResultsButton = vtkKWPushButton::New();
    this->SaveCurrentSelectedResultsButton->SetParent (curF);
    this->SaveCurrentSelectedResultsButton->Create();
    this->SaveCurrentSelectedResultsButton->SetImageToIcon ( this->QueryAtlasIcons->GetReserveSelectedURIsIcon() );
    this->SaveCurrentSelectedResultsButton->SetBorderWidth ( 0 );
    this->SaveCurrentSelectedResultsButton->SetReliefToFlat();    
    this->SaveCurrentSelectedResultsButton->SetBalloonHelpString ("Reserve selected results");


    vtkKWFrame *pastF = vtkKWFrame::New();
    pastF->SetParent ( managerFrame );
    pastF->Create();
    vtkKWFrame *toppastF = vtkKWFrame::New();
    toppastF->SetParent ( managerFrame );
    toppastF->Create();
    vtkKWLabel *pastL = vtkKWLabel::New();
    pastL->SetParent ( toppastF );
    pastL->Create();
    pastL->SetWidth ( 45 );
    pastL->SetText ( "Reserved search results" );
    pastL->SetBackgroundColor ( 0.85, 0.85, 0.95 );

    this->AccumulatedResultsList = vtkKWListBoxWithScrollbars::New();
    this->AccumulatedResultsList->SetParent ( toppastF );
    this->AccumulatedResultsList->Create();
    this->AccumulatedResultsList->GetWidget()->SetSelectionModeToMultiple();
    this->AccumulatedResultsList->GetWidget()->SetWidth ( 45 );
    this->AccumulatedResultsList->GetWidget()->SetHeight ( 4 );
    this->AccumulatedResultsList->HorizontalScrollbarVisibilityOn();
    this->AccumulatedResultsList->VerticalScrollbarVisibilityOn();
    this->AccumulatedResultsList->GetWidget()->SetDoubleClickCommand (this, "OpenLinkFromAccumulatedList");

    this->DeleteAccumulatedResultButton = vtkKWPushButton::New();
    this->DeleteAccumulatedResultButton->SetParent (pastF);
    this->DeleteAccumulatedResultButton->Create();
    this->DeleteAccumulatedResultButton->SetImageToIcon ( this->QueryAtlasIcons->GetClearSelectedIcon ( ) );
    this->DeleteAccumulatedResultButton->SetBorderWidth ( 0 );
    this->DeleteAccumulatedResultButton->SetReliefToFlat ( );    
    this->DeleteAccumulatedResultButton->SetBalloonHelpString ("Delete selected");

    this->DeselectAllAccumulatedResultsButton = vtkKWPushButton::New();
    this->DeselectAllAccumulatedResultsButton->SetParent (pastF);
    this->DeselectAllAccumulatedResultsButton->Create();
    this->DeselectAllAccumulatedResultsButton->SetImageToIcon ( this->QueryAtlasIcons->GetDeselectAllIcon() );
    this->DeselectAllAccumulatedResultsButton->SetBorderWidth ( 0 );
    this->DeselectAllAccumulatedResultsButton->SetReliefToFlat();    
    this->DeselectAllAccumulatedResultsButton->SetBalloonHelpString ( "Deselect all results");

    this->DeleteAllAccumulatedResultsButton = vtkKWPushButton::New();
    this->DeleteAllAccumulatedResultsButton->SetParent (pastF);
    this->DeleteAllAccumulatedResultsButton->Create();
    this->DeleteAllAccumulatedResultsButton->SetImageToIcon ( this->QueryAtlasIcons->GetClearAllIcon (  ) );
    this->DeleteAllAccumulatedResultsButton->SetBorderWidth ( 0 );
    this->DeleteAllAccumulatedResultsButton->SetReliefToFlat();    
    this->DeleteAllAccumulatedResultsButton->SetBalloonHelpString ("Delete all");

    this->SaveAccumulatedResultsButton = vtkKWLoadSaveButton::New();
    this->SaveAccumulatedResultsButton->SetParent (pastF);
    this->SaveAccumulatedResultsButton->Create();    
    this->SaveAccumulatedResultsButton->SetImageToIcon (  app->GetApplicationGUI()->GetApplicationToolbar()->GetSlicerToolbarIcons()->GetSaveSceneIcon() );   
    this->SaveAccumulatedResultsButton->SetBorderWidth ( 0 );
    this->SaveAccumulatedResultsButton->SetReliefToFlat();    
    this->SaveAccumulatedResultsButton->SetBalloonHelpString ("Save links to file");
    this->SaveAccumulatedResultsButton->SetCommand ( this, "WriteBookmarksCallback" );
    this->SaveAccumulatedResultsButton->GetLoadSaveDialog()->SetTitle("Save Firefox bookmarks file");
    this->SaveAccumulatedResultsButton->GetLoadSaveDialog()->ChooseDirectoryOff();
    this->SaveAccumulatedResultsButton->GetLoadSaveDialog()->SaveDialogOn();
    this->SaveAccumulatedResultsButton->GetLoadSaveDialog()->SetFileTypes ( "*.html");

    this->LoadURIsButton = vtkKWLoadSaveButton::New();
    this->LoadURIsButton->SetParent ( pastF);
    this->LoadURIsButton->Create();
    this->LoadURIsButton->SetImageToIcon ( app->GetApplicationGUI()->GetApplicationToolbar()->GetSlicerToolbarIcons()->GetLoadSceneIcon() );   
    this->LoadURIsButton->SetBorderWidth(0);
    this->LoadURIsButton->SetReliefToFlat ( );
    this->LoadURIsButton->SetBalloonHelpString ( "Load links from file" );
    this->LoadURIsButton->SetCommand ( this, "LoadBookmarksCallback" );
    this->LoadURIsButton->GetLoadSaveDialog()->SetTitle("Load Firefox bookmarks file");
    this->LoadURIsButton->GetLoadSaveDialog()->ChooseDirectoryOff();
    this->LoadURIsButton->GetLoadSaveDialog()->SaveDialogOff();
    this->LoadURIsButton->GetLoadSaveDialog()->SetFileTypes ( "*.html");

    app->Script ( "grid %s -row 1 -column 0 -columnspan 3 -pady 4 -padx 3", managerFrame->GetWidgetName() ); 

    app->Script( "pack %s -side top -padx 0 -pady 2 -fill both -expand 1", topcurF->GetWidgetName() );
    app->Script ("pack %s -side top -padx 0 -pady 2 -fill x -expand 1", curL->GetWidgetName() );
    app->Script ("pack %s -side top -padx 0 -pady 0 -fill both -expand 1", this->CurrentResultsList->GetWidgetName() );

    app->Script ("pack %s -side top -padx 0 -pady 2 -fill x -expand 1", curF->GetWidgetName() );
    app->Script ("grid %s -row 0 -column 0 -sticky ew -pady 4 -padx 3", this->DeselectAllCurrentResultsButton->GetWidgetName() );    
    app->Script ("grid %s -row 0 -column 1 -sticky ew -pady 4 -padx 3", this->DeleteCurrentResultButton->GetWidgetName() );    
    app->Script ("grid %s -row 0 -column 2 -sticky ew -pady 4 -padx 3", this->DeleteAllCurrentResultsButton->GetWidgetName() );    
    app->Script ("grid %s -row 0 -column 3 -sticky ew -pady 4 -padx 3", this->SaveCurrentSelectedResultsButton->GetWidgetName() );    
    app->Script ("grid %s -row 0 -column 4 -sticky ew -pady 4 -padx 3", this->SaveCurrentResultsButton->GetWidgetName() );    
    app->Script ("grid columnconfigure %s 0 -weight 1", this->DeselectAllCurrentResultsButton->GetWidgetName() );    
    app->Script ("grid columnconfigure %s 1 -weight 1", this->DeleteCurrentResultButton->GetWidgetName() );    
    app->Script ("grid columnconfigure %s 2 -weight 1", this->DeleteAllCurrentResultsButton->GetWidgetName() );    
    app->Script ("grid columnconfigure %s 3 -weight 1", this->SaveCurrentResultsButton->GetWidgetName() );    
    app->Script ("grid columnconfigure %s 4 -weight 1", this->SaveCurrentSelectedResultsButton->GetWidgetName() );    

    app->Script( "pack %s -side top -padx 0 -pady 2 -fill both -expand 1", toppastF->GetWidgetName() );
    app->Script ("pack %s -side top -padx 0 -pady 2 -fill x -expand 1", pastL->GetWidgetName() );
    app->Script ("pack %s -side top -padx 0 -pady 0 -fill both -expand 1", this->AccumulatedResultsList->GetWidgetName() );

    app->Script ("pack %s -side top -padx 0 -pady 2 -fill x -expand 1", pastF->GetWidgetName() );
    app->Script ("grid %s -row 0 -column 0 -sticky ew -pady 4 -padx 3", this->DeselectAllAccumulatedResultsButton->GetWidgetName() );    
    app->Script ("grid %s -row 0 -column 1 -sticky ew -pady 4 -padx 3", this->DeleteAccumulatedResultButton->GetWidgetName() );    
    app->Script ("grid %s -row 0 -column 2 -sticky ew -pady 4 -padx 3", this->DeleteAllAccumulatedResultsButton->GetWidgetName() );    
    app->Script ("grid %s -row 0 -column 3 -sticky ew -pady 4 -padx 3", this->LoadURIsButton->GetWidgetName() );    
    app->Script ("grid %s -row 0 -column 4 -sticky ew -pady 4 -padx 3", this->SaveAccumulatedResultsButton->GetWidgetName() );    
    app->Script ("grid columnconfigure %s 0 -weight 1", this->DeselectAllAccumulatedResultsButton->GetWidgetName() );    
    app->Script ("grid columnconfigure %s 1 -weight 1", this->DeleteAccumulatedResultButton->GetWidgetName() );    
    app->Script ("grid columnconfigure %s 2 -weight 1", this->DeleteAllAccumulatedResultsButton->GetWidgetName() );    
    app->Script ("grid columnconfigure %s 3 -weight 1", this->LoadURIsButton->GetWidgetName() );    
    app->Script ("grid columnconfigure %s 4 -weight 1", this->SaveAccumulatedResultsButton->GetWidgetName() );    

    curL->Delete();
    pastL->Delete();
    topcurF->Delete();
    curF->Delete();
    toppastF->Delete();
    pastF->Delete();
    managerFrame->Delete();

//    f->Delete();
    sl->Delete();
    searchFrame->Delete();
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::OpenLinkFromCurrentList ( )
{
  const char *url;
  
    url = this->CurrentResultsList->GetWidget()->GetSelection();
    //--- open in browser
    this->Script ( "QueryAtlasOpenLink %s", url);

}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::OpenLinkFromAccumulatedList ( )
{
  const char *url;
  
    url = this->AccumulatedResultsList->GetWidget()->GetSelection();
    //--- open in browser
    this->Script ( "QueryAtlasOpenLink %s", url );

}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildDisplayAndNavigationGUI ( )
{
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "QueryAtlas" );
    // -------------------------------------------------------------------------------------------------
    // ---
    // 3D DISPLAY AND NAVIGATION FRAME
    // ---
    // -------------------------------------------------------------------------------------------------
    vtkSlicerModuleCollapsibleFrame *displayFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    displayFrame->SetParent ( page );
    displayFrame->Create ( );
    displayFrame->SetLabelText ("3D Display & Navigation");
    displayFrame->CollapseFrame ( );
    // for now supress this frame.
    /*
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  displayFrame->GetWidgetName(),
                  this->UIPanel->GetPageWidget("QueryAtlas")->GetWidgetName());
    */
    displayFrame->Delete();
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildStructureFrame()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();

    // add multi-column list box for search terms
    // IDEA: click 'add term' button and get a new row in this widget.
    // widget (for now) had two columns, first has a radiobutton in it
    // for marking the row for use in search, second is an entry widget
    // in which user types a search term.
    // Delete button above works when a row is highlighted. how is
    // that accomplished? maybe a better way. Just delete rows with
    // radio button selected? (select for use and for deletion?)
    // grrr. i don't yet understand how this widget works.
    this->StructureListWidget = vtkQueryAtlasUseSearchTermWidget::New ( );
    this->StructureListWidget->SetParent ( this->StructureFrame );
    this->StructureListWidget->Create ( );
//    int i = this->StructureListWidget->GetMultiColumnList()->GetWidget()->GetColumnIndexWithName ( "Search terms" );
//    this->StructureListWidget->GetMultiColumnList()->GetWidget()->SetColumnName ( i, "Structure terms");
    app->Script ( "pack %s -side top -fill x -expand true", this->StructureListWidget->GetWidgetName() );
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildSpeciesFrame()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();

    this->SpeciesLabel = vtkKWLabel::New();
    this->SpeciesLabel->SetParent ( this->SpeciesFrame );
    this->SpeciesLabel->Create();
    this->SpeciesLabel->SetText( "species to include: ");
    
    this->SpeciesNoneButton = vtkKWRadioButton::New();
    this->SpeciesNoneButton->SetParent ( this->SpeciesFrame);
    this->SpeciesNoneButton->Create();
    this->SpeciesNoneButton->SetValue ("n/a");
    this->SpeciesNoneButton->SetText ("don't specify");
    this->SpeciesNoneButton->SetSelectedState ( 1 );
    
    this->SpeciesHumanButton = vtkKWRadioButton::New();
    this->SpeciesHumanButton->SetParent ( this->SpeciesFrame);
    this->SpeciesHumanButton->Create();
    this->SpeciesHumanButton->SetValue ("human");
    this->SpeciesHumanButton->SetText ("human");
    this->SpeciesHumanButton->SetVariableName ( this->SpeciesNoneButton->GetVariableName() );
    
    this->SpeciesMouseButton = vtkKWRadioButton::New();
    this->SpeciesMouseButton->SetParent ( this->SpeciesFrame );
    this->SpeciesMouseButton->Create();
    this->SpeciesMouseButton->SetText("mouse");
    this->SpeciesMouseButton->SetValue ("mouse");
    this->SpeciesMouseButton->SetVariableName ( this->SpeciesNoneButton->GetVariableName() );

    this->SpeciesMacaqueButton = vtkKWRadioButton::New();
    this->SpeciesMacaqueButton->SetParent ( this->SpeciesFrame);
    this->SpeciesMacaqueButton->Create();
    this->SpeciesMacaqueButton->SetText ("macaque");
    this->SpeciesMacaqueButton->SetValue ("macaque");
    this->SpeciesMacaqueButton->SetVariableName ( this->SpeciesNoneButton->GetVariableName() );

    app->Script ( "grid %s -row 0 -column 0 -sticky w", this->SpeciesLabel->GetWidgetName() );
    app->Script ( "grid %s -row 0 -column 1 -sticky w", this->SpeciesNoneButton->GetWidgetName() );
    app->Script ( "grid %s -row 1 -column 1 -sticky w", this->SpeciesHumanButton->GetWidgetName() );
    app->Script ( "grid %s -row 2 -column 1 -sticky w", this->SpeciesMouseButton->GetWidgetName() );
    app->Script ( "grid %s -row 3 -column 1 -sticky w", this->SpeciesMacaqueButton->GetWidgetName() );
    
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildPopulationFrame()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();

    this->DiagnosisMenuButton = vtkKWMenuButtonWithLabel::New();
    this->DiagnosisMenuButton->SetParent ( this->PopulationFrame );
    this->DiagnosisMenuButton->Create ();
    this->DiagnosisMenuButton->SetLabelText ( "diagnoses: ");
    this->DiagnosisMenuButton->SetLabelPositionToLeft ( );
    this->DiagnosisMenuButton->GetLabel()->SetWidth (12);
    this->DiagnosisMenuButton->GetWidget()->SetWidth (28);
    this->BuildDiagnosisMenu ( this->DiagnosisMenuButton->GetWidget()->GetMenu() );
    this->DiagnosisMenuButton->GetWidget()->GetMenu()->AddSeparator();
    this->DiagnosisMenuButton->GetWidget()->GetMenu()->AddCommand ( "close" );

    this->AddDiagnosisEntry = vtkKWEntryWithLabel::New();
    this->AddDiagnosisEntry->SetParent ( this->PopulationFrame );
    this->AddDiagnosisEntry->Create();
    this->AddDiagnosisEntry->SetLabelText ( "add diagnosis: " );
    this->AddDiagnosisEntry->SetLabelPositionToLeft ( );
    this->AddDiagnosisEntry->GetLabel()->SetWidth(12);
    this->AddDiagnosisEntry->GetWidget()->SetWidth (35 );

    this->GenderMenuButton = vtkKWMenuButtonWithLabel::New();
    this->GenderMenuButton->SetParent ( this->PopulationFrame );
    this->GenderMenuButton->Create ();
    this->GenderMenuButton->SetLabelText ( "gender: ");
    this->GenderMenuButton->GetWidget()->SetWidth (28);
    this->GenderMenuButton->SetLabelPositionToLeft ( );
    this->GenderMenuButton->GetLabel()->SetWidth ( 12);
    this->GenderMenuButton->GetWidget()->GetMenu()->AddRadioButton ("n/a");
    this->GenderMenuButton->GetWidget()->GetMenu()->SelectItem ("n/a");
    this->GenderMenuButton->GetWidget()->GetMenu()->AddRadioButton ("M");
    this->GenderMenuButton->GetWidget()->GetMenu()->AddRadioButton ("F");
    this->GenderMenuButton->GetWidget()->GetMenu()->AddRadioButton ("mixed");
    this->GenderMenuButton->GetWidget()->GetMenu()->AddSeparator();
    this->GenderMenuButton->GetWidget()->GetMenu()->AddCommand ( "close" );    

    this->HandednessMenuButton = vtkKWMenuButtonWithLabel::New();
    this->HandednessMenuButton->SetParent ( this->PopulationFrame );
    this->HandednessMenuButton->Create ();
    this->HandednessMenuButton->SetLabelText ( "handedness: ");
    this->HandednessMenuButton->GetWidget()->SetWidth (28);
    this->HandednessMenuButton->GetLabel()->SetWidth (12);
    this->HandednessMenuButton->SetLabelPositionToLeft ( );
    this->HandednessMenuButton->GetWidget()->GetMenu()->AddRadioButton ("n/a");
    this->HandednessMenuButton->GetWidget()->GetMenu()->SelectItem ("n/a");
    this->HandednessMenuButton->GetWidget()->GetMenu()->AddRadioButton ("left");
    this->HandednessMenuButton->GetWidget()->GetMenu()->AddRadioButton ("right");
    this->HandednessMenuButton->GetWidget()->GetMenu()->AddRadioButton ("mixed");
    this->HandednessMenuButton->GetWidget()->GetMenu()->AddSeparator();
    this->HandednessMenuButton->GetWidget()->GetMenu()->AddCommand ("close");

    this->AgeMenuButton = vtkKWMenuButtonWithLabel::New();
    this->AgeMenuButton->SetParent ( this->PopulationFrame );
    this->AgeMenuButton->Create ();
    this->AgeMenuButton->SetLabelText ( "age ranges: ");
    this->AgeMenuButton->GetWidget()->SetWidth (28);
    this->AgeMenuButton->GetLabel()->SetWidth (12);
    this->AgeMenuButton->SetLabelPositionToLeft ( );
    this->AgeMenuButton->GetWidget()->GetMenu()->AddRadioButton ("n/a");
    this->AgeMenuButton->GetWidget()->GetMenu()->SelectItem ("n/a");
    this->AgeMenuButton->GetWidget()->GetMenu()->AddRadioButton ("neonate");
    this->AgeMenuButton->GetWidget()->GetMenu()->AddRadioButton ("infant");
    this->AgeMenuButton->GetWidget()->GetMenu()->AddRadioButton ("child");
    this->AgeMenuButton->GetWidget()->GetMenu()->AddRadioButton ("adolescent");
    this->AgeMenuButton->GetWidget()->GetMenu()->AddRadioButton ("adult");
    this->AgeMenuButton->GetWidget()->GetMenu()->AddRadioButton ("elderly");
    this->AgeMenuButton->GetWidget()->GetMenu()->AddSeparator();
    this->AgeMenuButton->GetWidget()->GetMenu()->AddCommand ( "close");    

    app->Script ( "pack %s %s %s %s %s -side top -padx 5 -pady 2 -anchor nw",
                  this->DiagnosisMenuButton->GetWidgetName(),
                  this->AddDiagnosisEntry->GetWidgetName(),
                  this->GenderMenuButton->GetWidgetName (),
                  this->HandednessMenuButton->GetWidgetName(),
                  this->AgeMenuButton->GetWidgetName());
}




//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::UnpackLoaderContextFrames ( )
{
    this->Script ( "pack forget %s", this->FIPSFSFrame->GetWidgetName() );
    this->Script ( "pack forget %s", this->QdecFrame->GetWidgetName() );
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::PackLoaderContextFrame ( vtkKWFrame *f )
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  app->Script ( "pack %s -side top -anchor nw -expand 0 -fill x -padx 2 -pady 0", f->GetWidgetName( ));
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildLoaderContextButtons ( vtkKWFrame *parent )
{
  vtkKWFrame *f = vtkKWFrame::New();
  f->SetParent ( parent );
  f->Create();
  this->Script ("pack %s -side top -anchor nw -fill x -expand n", f->GetWidgetName() );

  this->FIPSFSButton = vtkKWPushButton::New();
  this->FIPSFSButton->SetParent ( f );
  this->FIPSFSButton->Create();
  this->FIPSFSButton->SetWidth ( 12 );
  this->FIPSFSButton->SetText ( "FIPS+FreeSurfer" );

  this->QdecButton = vtkKWPushButton::New();
  this->QdecButton->SetParent ( f );
  this->QdecButton->Create();
  this->QdecButton->SetWidth ( 12 );
  this->QdecButton->SetText ( "Qdec");

  this->Script ( "pack %s %s -anchor nw -side left -fill none -padx 2 -pady 0",
                 this->FIPSFSButton->GetWidgetName(),
                 this->QdecButton->GetWidgetName() );


  f->Delete();
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildLoaderContextFrames ( vtkKWFrame *parent )
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    
    this->FIPSFSFrame = vtkKWFrame::New();
    this->FIPSFSFrame->SetParent ( parent );
    this->FIPSFSFrame->Create();
    this->FIPSFSFrame->SetReliefToGroove();
    this->FIPSFSFrame->SetBorderWidth ( 1 );
    
    this->QdecFrame = vtkKWFrame::New();
    this->QdecFrame->SetParent ( parent );
    this->QdecFrame->Create();
    this->QdecFrame->SetReliefToGroove();
    this->QdecFrame->SetBorderWidth ( 1 );
}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::ColorCodeLoaderContextButtons ( vtkKWPushButton *b )
{
  this->FIPSFSButton->SetBackgroundColor ( _br, _bg, _bb );
  this->QdecButton->SetBackgroundColor ( _br, _bg, _bb );

  this->FIPSFSButton->SetForegroundColor ( _fr, _fg, _fb );
  this->QdecButton->SetForegroundColor ( _fr, _fg, _fb );

  b->SetBackgroundColor (1.0, 1.0, 1.0);
  b->SetForegroundColor (0.0, 0.0, 0.0);
}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildOtherFrame()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();

    // add multi-column list box for search terms

    this->OtherListWidget = vtkQueryAtlasUseSearchTermWidget::New();
    this->OtherListWidget->SetParent ( this->OtherFrame );
    this->OtherListWidget->Create ( );
    this->OtherListWidget->GetMultiColumnList()->GetWidget()->SetHeight(3);
//    int i = this->OtherListWidget->GetMultiColumnList()->GetWidget()->GetColumnIndexWithName ( "Search terms" );
//    this->OtherListWidget->GetMultiColumnList()->GetWidget()->SetColumnName ( i, "Other search terms");
    app->Script ( "pack %s -side top -fill x -expand true", this->OtherListWidget->GetWidgetName() );

}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::UnpackQueryBuilderContextFrames ( )
{
    this->Script ( "pack forget %s", this->OtherFrame->GetWidgetName() );
    this->Script ( "pack forget %s", this->StructureFrame->GetWidgetName() );
    this->Script ( "pack forget %s", this->PopulationFrame->GetWidgetName() );
    this->Script ( "pack forget %s", this->SpeciesFrame->GetWidgetName() );
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::PackQueryBuilderContextFrame ( vtkKWFrame *f )
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  app->Script ( "pack %s -side top -anchor nw -expand 0 -fill x", f->GetWidgetName( ));
}

//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildQueryBuilderContextFrames ( vtkKWFrame *parent )
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    
    this->StructureFrame = vtkKWFrame::New();
    this->StructureFrame->SetParent ( parent );
    this->StructureFrame->Create();
    
    this->PopulationFrame = vtkKWFrame::New();
    this->PopulationFrame->SetParent ( parent );
    this->PopulationFrame->Create();

    this->SpeciesFrame = vtkKWFrame::New();
    this->SpeciesFrame->SetParent ( parent );
    this->SpeciesFrame->Create();

    this->OtherFrame = vtkKWFrame::New();
    this->OtherFrame->SetParent ( parent );
    this->OtherFrame->Create();

}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildQueryBuilderContextButtons ( vtkKWFrame *parent )
{
  vtkKWFrame *f = vtkKWFrame::New();
  f->SetParent ( parent );
  f->Create();
  this->Script ("pack %s -side top -anchor nw -fill none", f->GetWidgetName() );

  // for now this will be the "other" term repository...
  // when we flesh this out with multiscale categories,
  // 
  this->OtherButton = vtkKWPushButton::New();
  this->OtherButton->SetParent ( f );
  this->OtherButton->Create();
  this->OtherButton->SetWidth ( 10 );
  this->OtherButton->SetText ( "other");
    
  this->StructureButton = vtkKWPushButton::New();
  this->StructureButton->SetParent ( f );
  this->StructureButton->Create();
  this->StructureButton->SetWidth ( 10);
  this->StructureButton->SetText ( "structure");
    
  this->PopulationButton = vtkKWPushButton::New();
  this->PopulationButton->SetParent ( f );
  this->PopulationButton->Create();
  this->PopulationButton->SetWidth ( 10 );
  this->PopulationButton->SetText ( "group");
    
  this->SpeciesButton = vtkKWPushButton::New();
  this->SpeciesButton->SetParent ( f );
  this->SpeciesButton->Create();
  this->SpeciesButton->SetWidth ( 10 );
  this->SpeciesButton->SetText ( "species");    

  this->Script ( "pack %s %s %s %s -anchor nw -side left -fill none -padx 2 -pady 2",
                 this->OtherButton->GetWidgetName(),
                 this->StructureButton->GetWidgetName(),
                 this->PopulationButton->GetWidgetName(),
                 this->SpeciesButton->GetWidgetName() );

  f->Delete();
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildDatabasesMenu ( vtkKWMenu *m )
{
  m->AddRadioButton ("all");
  m->SelectItem ("all");
  m->AddSeparator();
  m->AddRadioButton ("Google");
  m->AddRadioButton ("Wikipedia");
  m->AddSeparator();
  m->AddRadioButton ("PubMed");
  m->AddRadioButton ("JNeurosci");
  m->AddRadioButton ("PLoS");
  m->AddSeparator();
  m->AddRadioButton ("Metasearch");
  m->AddRadioButton ("Entrez");
  m->AddSeparator();
  m->AddRadioButton ("IBVD");
  m->AddRadioButton ("BrainInfo");
  m->AddSeparator();
  m->AddCommand ( "close");
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::BuildDiagnosisMenu( vtkKWMenu *m )
{
  m->DeleteAllItems( );
  m->AddRadioButton ( "Normal" );
  m->SelectItem ("Normal");
  m->AddRadioButton ("Alzheimer's Disease");
  m->AddRadioButton ("Schizophrenia");
  m->AddRadioButton  ("Alcoholism");
  m->AddRadioButton  ("Dementia");
  m->AddRadioButton  ("Autism");
  m->AddRadioButton ( "Depression");
  m->AddRadioButton ("Traumatic Brain Injury");
  m->AddRadioButton ("OCD");
  m->AddRadioButton ("ADHD");
  m->AddRadioButton ("Epilepsy");
  m->AddRadioButton ("PDAPP Transgenic");
}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::AddToDiagnosisMenu ( vtkKWMenu *m, const char *diagnosis )
{
  this->BuildDiagnosisMenu ( m );
  m->AddRadioButton ( diagnosis );
  m->SelectItem ( diagnosis );
}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::GetDiagnosisTerms ( )
{

  const char *term;
  
  this->DiagnosisTerms.clear();
  //-- get diagnosis
  term = this->GetDiagnosisMenuButton()->GetWidget()->GetValue();
  this->DiagnosisTerms.push_back ( std::string(term) );

  //-- get gender
  term = this->GetGenderMenuButton()->GetWidget()->GetValue();
  this->DiagnosisTerms.push_back ( std::string(term) );

  //-- get age
  term = this->GetAgeMenuButton()->GetWidget()->GetValue();
  this->DiagnosisTerms.push_back ( std::string(term) );
  
  //-- get handedness
  term = this->GetHandednessMenuButton()->GetWidget()->GetValue();
  this->DiagnosisTerms.push_back ( std::string(term) );

}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::GetSpeciesTerms ( )
{
  this->SpeciesTerms.clear();
  this->SpeciesTerms.push_back ( std::string ( this->SpeciesNoneButton->GetVariableValue() ));
}


//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::GetStructureTerms ( )
{

  this->StructureTerms.clear();
  // this counts the number of selected items instead of the number of CHECKED items
  int num = this->StructureListWidget->GetNumberOfSearchTermsToUse();
  for ( int i = 0; i < num; i++ )
    {
    this->StructureTerms.push_back ( std::string (this->StructureListWidget->GetNthSearchTermToUse ( i ) ) );
    }
}



//---------------------------------------------------------------------------
void vtkQueryAtlasGUI::GetOtherTerms ( )
{

  this->OtherTerms.clear();
  int num = this->StructureListWidget->GetNumberOfSearchTermsToUse();
  for ( int i = 0; i < num; i++ )
    {
    this->StructureTerms.push_back ( std::string (this->StructureListWidget->GetNthSearchTermToUse ( i ) ) );
    }

}
