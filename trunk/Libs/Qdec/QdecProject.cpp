/**
 * @file  QdecProject.cpp
 * @brief API class containing all qdec subject data and methods
 *
 * Top-level interface class containing all data associated with a users
 * subject group, and potentially mri_glmfit processed data associated with
 * that group.
 */
/*
 * Original Author: Nick Schmansky
 * CVS Revision Info:
 *    $Author: nicks $
 *    $Date: 2007/05/23 21:20:58 $
 *    $Revision: 1.5 $
 *
 * Copyright (C) 2007,
 * The General Hospital Corporation (Boston, MA).
 * All rights reserved.
 *
 * Distribution, usage and copying of this software is covered under the
 * terms found in the License Agreement file named 'COPYING' found in the
 * FreeSurfer source code root directory, and duplicated here:
 * https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferOpenSourceLicense
 *
 * General inquiries: freesurfer@nmr.mgh.harvard.edu
 * Bug reports: analysis-bugs@nmr.mgh.harvard.edu
 *
 */

#include <stdexcept>
#include <sstream>
#include <iostream>
#include <fstream>

#include "QdecProject.h"


// Constructors/Destructors
//

QdecProject::QdecProject ( )
{
  this->mfnProjectFile = "qdec.xml";
  this->mDataTable = new QdecDataTable();
  this->mGlmDesign = new QdecGlmDesign( this->mDataTable );
  this->mGlmFitter = new QdecGlmFit();
}

QdecProject::~QdecProject ( )
{
  delete this->mDataTable;
  delete this->mGlmDesign;
  delete this->mGlmFitter;
}

//
// Methods
//

/**
 * Load a .qdec project file (containing all necessary info to begin
 * working either on a new project, or to continue working using
 * results from a prior saved work session). isDataDir should be a
 * directory where we can expand the .qdec file (like /tmp).
 * @return int
 * @param  isFileName
 * @param  isDataDir
 */
int QdecProject::LoadProjectFile ( const char* ifnProject,
                                   const char* ifnDataDir )
{

  string fnProject( ifnProject );

  // Find the base name of the project file.
  string fnProjectBase( ifnProject );
  string::size_type nPreLastSlash = fnProject.rfind( '/' );
  if( string::npos != nPreLastSlash )
    fnProjectBase = fnProject.substr( nPreLastSlash+1, fnProject.size() );
  
  // Make a target dir for the expanded file in the data dir, with a
  // directory name of the project file.
  string fnExpandedProjectDir = string(ifnDataDir) + "/" + fnProjectBase + ".working";

  string sSubject;
  string sHemisphere;
  string sAnalysisName;
  string fnDataTableBase;
  string sDiscreteFactor1 = "none";
  string sDiscreteFactor2 = "none";
  string sContinuousFactor1 = "none";
  string sContinuousFactor2 = "none";
  string sMeasure;
  int smoothness = -1;

  // Check the file.
  ifstream fInput( ifnProject, std::ios::in );
  if( !fInput || fInput.bad() )
    {
    throw runtime_error( string("Couldn't open file " ) + ifnProject );
    }
  fInput.close();

  // Erase old working directory if present.
  string sCommand = "rm -rf " + fnExpandedProjectDir;
  int rSystem = system( sCommand.c_str() );
  if( 0 != rSystem )
    {
    fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Couldn't "
             "remove existing temp directory (cmd=%s)\n", sCommand.c_str() );
    return -1;
    }
  // Expand the .qdec file into the destination directory.
  sCommand = string("cd ") + ifnDataDir + "; "
    "tar zxvf " + ifnProject + " > /dev/null";
  rSystem = system( sCommand.c_str() );
  if( 0 != rSystem ) {
    fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Couldn't "
             "untar project file (cmd=%s)\n", sCommand.c_str() );
    return -1;
  }

  // Look for and check the version file.
  string fnVersion = fnExpandedProjectDir + "/Version.txt";
  ifstream fVersion( fnVersion.c_str(), ios::out );
  if( !fVersion || fVersion.bad() ) {
    fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Couldn't "
             "find Version file\n" );
    return -1;
  }
  int version;
  fVersion >> version;
  fVersion.close();
  if( 1 != version ) {
    fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Version "
             "file had wrong value (%d)\n", version );
    return -1;
  }
  
  // Parse the meta data file.
  string fnMetadata = fnExpandedProjectDir + "/" + this->GetMetadataFileName();
  ifstream fMetadata( fnMetadata.c_str(), ios::in );
  if( !fMetadata || fMetadata.bad() ) {
    fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Couldn't "
             "open metadata file %s\n", fnMetadata.c_str() );
    return -1;
  }
  // Make sure the first token is QdecProjectMetadata, and then the
  // next line is Version 1.
  string sToken;
  string asCorrectTokens[] = { "QdecProjectMetadata", "Version", "1" };
  for( int nToken = 0; nToken < 3; nToken++ ) {
    fMetadata >> sToken;
    if( sToken != asCorrectTokens[nToken] ) {
      fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Invalid "
               "metadata file %s, %s token not found\n", 
               fnMetadata.c_str(), asCorrectTokens[nToken].c_str() );
      return -1;
    }
  }
  // Now we parse the file and look for names and values.
  while( fMetadata >> sToken && !fMetadata.eof() ) {
    if( sToken == "Subject" ) fMetadata >> sSubject;
    else if( sToken == "Hemisphere" ) fMetadata >> sHemisphere;
    else if( sToken == "AnalysisName" ) fMetadata >> sAnalysisName;
    else if( sToken == "DataTable" ) fMetadata >> fnDataTableBase;
    else if( sToken == "Measure" ) fMetadata >> sMeasure;
    else if( sToken == "Smoothness" ) fMetadata >> smoothness;
    else if( sToken == "DiscreteFactor1" ) fMetadata >> sDiscreteFactor1;
    else if( sToken == "DiscreteFactor2" ) fMetadata >> sDiscreteFactor2;
    else if( sToken == "ContinuousFactor1" )fMetadata >> sContinuousFactor1;
    else if( sToken == "ContinuousFactor2" )fMetadata >> sContinuousFactor2;
    else {
      fprintf( stderr, "WARNING: QdecProject::LoadProjectFile: Unrecognized "
               "token in QdecProjectMetadata: %s\n", sToken.c_str() );
    }
  }

  // Make sure we got some decent results.
  if( sSubject == "" ) {
      fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Invalid "
               "project metadata file, Subject value not found\n" );
      return -1;
  }
  if( sHemisphere == "" ) {
      fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Invalid "
               "project metadata file, Hemisphere value not found\n" );
      return -1;
  }
  if( sAnalysisName == "" ) {
      fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Invalid "
               "project metadata file, AnalysisName value not found\n" );
      return -1;
  }
  if( fnDataTableBase == "" ) {
      fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Invalid "
               "project metadata file, DataTable value not found\n" );
      return -1;
  }
  if( sMeasure == "" ) {
      fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Invalid "
               "project metadata file, Measure value not found\n" );
      return -1;
  }
  if( -1 == smoothness ) {
      fprintf( stderr, "ERROR: QdecProject::LoadProjectFile: Invalid "
               "project metadata file, Smoothness value not found\n" );
      return -1;
  }
  
  // Load our data table. Note that this might set the subjects dir,
  // but we'll set it later to our data dir.
  string fnDataTable = fnExpandedProjectDir + "/" + fnDataTableBase;
  int errorCode;
  errorCode = this->LoadDataTable( fnDataTable.c_str() );
  if( errorCode )
    return errorCode;

  // Set the subjects dir to the data dir, so that we can find the
  // subject.
  this->SetSubjectsDir( fnExpandedProjectDir.c_str() );

  // Set the working dir to isDataDir/sAnalysisName.
  string fnWorkingDir = fnExpandedProjectDir + "/" + sAnalysisName;
  this->SetWorkingDir( fnWorkingDir.c_str() );

  // We're generating design and results here so that we can access it
  // as metadata, but we're not actually computing any new results;
  // those all exist in our data dir.

  // Create the design. This will be used in the results.
  errorCode = 
    this->mGlmDesign->Create ( this->mDataTable,
                               sAnalysisName.c_str(),
                               sDiscreteFactor1.c_str(),
                               sDiscreteFactor2.c_str(),
                               sContinuousFactor1.c_str(),
                               sContinuousFactor2.c_str(),
                               sMeasure.c_str(),
                               sHemisphere.c_str(),
                               smoothness,
                               NULL );
  if( errorCode )
    return errorCode;
  
  // Create fit results data.
  errorCode = 
    mGlmFitter->CreateResultsFromCachedData ( this->mGlmDesign );
  if( errorCode )
    return errorCode;

  return 0;
}



/**
 * Save all necessary information pertaining to this project (all subject
 * data, any results, any user preferences).
 * @return int
 * @param  isFileName
 */
int QdecProject::SaveProjectFile ( const char* ifnProject,
                                   const char* ifnDataDir )
{
  /* To make our file, we create a temp directory, link in our files,
     and then tar it up into the destination .qdec file. This is the
     structure we want. We'll create symlinks and then tar it up into
     the destination file.

    $ifnWorkingDir/$project.qdec.working/
                               $Subject/surf/{r,l}h.{curv,inflatd,pial,white}
                                        label/{r,l}h.aparc.annot
                               $AnalysisName/ *
                               qdec.table.dat
                               QdecProjectMetadata.txt
  */

  string fnSubjectsDir = this->GetSubjectsDir();
  string sSubjectName = this->GetAverageSubject();
  string fnWorkingDir = this->GetWorkingDir();
  string fnProject( ifnProject );

  // Find the base name of the project file.
  string fnProjectBase( ifnProject );
  string::size_type nPreLastSlash = fnProject.rfind( '/' );
  if( string::npos != nPreLastSlash )
    fnProjectBase = fnProject.substr( nPreLastSlash+1, fnProject.size() );

  // Make a target dir for the expanded file in the data dir, with a
  // directory name of the project file.
  string fnExpandedProjectDir = string(ifnDataDir) + "/" + fnProjectBase + ".working";

  // Erase old working directory if present.
  string sCommand = "rm -rf " + fnExpandedProjectDir;
  int rSystem = system( sCommand.c_str() );
  if( 0 != rSystem ) {
    fprintf( stderr, "ERROR: QdecProject::SaveProjectFile: Couldn't "
             "remove existing temp directory (cmd=%s)\n", sCommand.c_str() );
    return -1;
  }

  // Make a temporary directory for our data.
  sCommand = "mkdir " + fnExpandedProjectDir;
  rSystem = system( sCommand.c_str() );
  if( 0 != rSystem ) {
    fprintf( stderr, "ERROR: QdecProject::SaveProjectFile: Couldn't "
             "make temp directory (cmd=%s)\n", sCommand.c_str() );
    return -1;
  }

  // Write a version file to it.
  string fnVersion = fnExpandedProjectDir + "/Version.txt";
  ofstream fVersion( fnVersion.c_str(), ios::out );
  fVersion << "1" << endl;
  fVersion.close();

  // Make the average subject dir structure.
  sCommand = "mkdir -p " + 
    fnExpandedProjectDir + "/" + sSubjectName + "/surf " + 
    fnExpandedProjectDir + "/" + sSubjectName + "/label";
  rSystem = system( sCommand.c_str() );
  if( 0 != rSystem ) {
    fprintf( stderr, "ERROR: QdecProject::SaveProjectFile: Couldn't "
             "make subject dir structure (cmd=%s)\n", sCommand.c_str() );
    return -1;
  }
  
  // Link the necessary files. Start with surfaces.
  sCommand = "ln -s " +
    fnSubjectsDir + "/" + sSubjectName + "/surf/*.curv " +
    fnSubjectsDir + "/" + sSubjectName + "/surf/*.inflated " +
    fnSubjectsDir + "/" + sSubjectName + "/surf/*.pial " +
    fnSubjectsDir + "/" + sSubjectName + "/surf/*.white " +
    fnExpandedProjectDir + "/" + sSubjectName + "/surf";
  rSystem = system( sCommand.c_str() );
  if( 0 != rSystem ) {
    fprintf( stderr, "ERROR: QdecProject::SaveProjectFile: Couldn't "
             "link surface files (cmd=%s)\n", sCommand.c_str() );
    return -1;
  }
  
  // Annotations.
  sCommand = "ln -s " +
    fnSubjectsDir + "/" + sSubjectName + "/label/*.aparc.annot " +
    fnExpandedProjectDir + "/" + sSubjectName + "/label";
  rSystem = system( sCommand.c_str() );
  if( 0 != rSystem ) {
    fprintf( stderr, "ERROR: QdecProject::SaveProjectFile: Couldn't "
             "link annotation files (cmd=%s)\n", sCommand.c_str() );
    return -1;
  }

  // The whole working dir.
  sCommand = "ln -s " + fnWorkingDir + " " + fnExpandedProjectDir;
  rSystem = system( sCommand.c_str() );
  if( 0 != rSystem ) {
    fprintf( stderr, "ERROR: QdecProject::SaveProjectFile: Couldn't "
             "link working dir (cmd=%s)\n", sCommand.c_str() );
    return -1;
  }

  // Data table.
  string fnDataTable = this->GetDataTable()->GetFileName();
  string fnDataTablePath( fnDataTablePath );
  string fnDataTableBase( fnDataTable );
  nPreLastSlash = fnDataTable.rfind( '/' );
  if( string::npos != nPreLastSlash ) {
    fnDataTableBase = fnDataTable.substr( nPreLastSlash+1, fnDataTable.size());
    fnDataTablePath = fnDataTable.substr( 0, nPreLastSlash+1);
  }
  sCommand = "ln -s " + fnDataTable + " " +
    fnDataTablePath + "/*.levels " + fnExpandedProjectDir;
  rSystem = system( sCommand.c_str() );
  if( 0 != rSystem ) {
    fprintf( stderr, "ERROR: QdecProject::SaveProjectFile: Couldn't "
             "link data table (cmd=%s)\n", sCommand.c_str() );
    return -1;
  }

  // Generate the meta data file.
  string fnMetadata = fnExpandedProjectDir + "/" + this->GetMetadataFileName();
  ofstream fMetadata( fnMetadata.c_str(), ios::out );
  if( !fMetadata || fMetadata.bad() ) {
    fprintf( stderr, "ERROR: QdecProject::SaveProjectFile: Couldn't "
             "make metadata file %s\n", fnMetadata.c_str() );
    return -1;
  }
  fMetadata << "QdecProjectMetadata" << endl;
  fMetadata << "Version 1" << endl;
  fMetadata << "Subject " << this->GetAverageSubject() << endl;
  fMetadata << "Hemisphere " << this->GetHemi() << endl;
  fMetadata << "AnalysisName " << this->GetGlmDesign()->GetName() << endl;
  fMetadata << "DataTable " << fnDataTableBase << endl;
  fMetadata << "Measure " << this->GetGlmDesign()->GetMeasure() << endl;
  fMetadata << "Smoothness " << this->GetGlmDesign()->GetSmoothness() << endl;
  
  // We only support the two factors of each kind, so get the vectors
  // and just write the first and second ones if they are present.
  vector<QdecFactor*> const& lDiscreteFactors =
    this->GetGlmDesign()->GetDiscreteFactors();
  if( lDiscreteFactors.size() > 0 )
    fMetadata << "DiscreteFactor1 " 
              << lDiscreteFactors[0]->GetFactorName() << endl;
  if( lDiscreteFactors.size() > 1 )
    fMetadata << "DiscreteFactor2 "
              << lDiscreteFactors[1]->GetFactorName() << endl;

  vector<QdecFactor*> const& lContinuousFactors =
    this->GetGlmDesign()->GetContinuousFactors();
  if( lContinuousFactors.size() > 0 )
    fMetadata << "ContinuousFactor1 " 
              << lContinuousFactors[0]->GetFactorName() << endl;
  if( lContinuousFactors.size() > 1 )
    fMetadata << "ContinuousFactor2 " 
              << lContinuousFactors[1]->GetFactorName() << endl;

  fMetadata.close();


  // Tar them up to the destination location with the .qdec filename.
  sCommand = string("cd ") + ifnDataDir + "; " +
    "tar hcfzv " + ifnProject + " " + fnProjectBase + ".working > /dev/null";
  rSystem = system( sCommand.c_str() );
  if( 0 != rSystem ) {
    fprintf( stderr, "ERROR: QdecProject::SaveProjectFile: Couldn't "
             "tar project table (cmd=%s)\n", sCommand.c_str() );
    return -1;
  }

  // Delete the temp directory.
  sCommand = "rm -rf " + fnExpandedProjectDir;
  rSystem = system( sCommand.c_str() );
  if( 0 != rSystem ) {
    fprintf( stderr, "ERROR: QdecProject::SaveProjectFile: Couldn't "
             "remove temp directory (cmd=%s)\n", sCommand.c_str() );
    return -1;
  }

  return 0;
}


/**
 * @return int
 * @param  isFileName
 */
int QdecProject::LoadDataTable ( const char* isFileName )
{
  char sd[3000];
  int ret = this->mDataTable->Load ( isFileName, sd );
  if ( ret )
    {
    return ret;
    }
  if ( strlen(sd) > 0 )
    {
    ret = this->SetSubjectsDir ( sd );
    }
  return ret;
}


/**
 * @return void
 * @param  iFilePointer
 */
void QdecProject::DumpDataTable ( FILE* iFilePointer )
{
  return this->mDataTable->Dump ( iFilePointer );
}

/**
 * @return int
 * @param  isFileName
 */
int QdecProject::SaveDataTable ( const char* isFileName )
{
  return this->mDataTable->Save ( isFileName );
}


/**
 * @return QdecDataTable*
 */
QdecDataTable* QdecProject::GetDataTable ( )
{
  return this->mDataTable;
}


/**
 * @return string
 */
string QdecProject::GetSubjectsDir ( )
{
  return this->mGlmDesign->GetSubjectsDir();
}


/**
 * @param  ifnSubjectsDir
 */
int QdecProject::SetSubjectsDir ( const char* ifnSubjectsDir )
{
  return this->mGlmDesign->SetSubjectsDir( ifnSubjectsDir );
}


/**
 * @return string
 */
string QdecProject::GetAverageSubject ( )
{
  return this->mGlmDesign->GetAverageSubject();
}


/**
 * @param  isSubjectName
 */
void QdecProject::SetAverageSubject ( const char* isSubjectName )
{
  this->mGlmDesign->SetAverageSubject( isSubjectName );
}

/**
 * @return string
 */
string QdecProject::GetDefaultWorkingDir ( )
{
  return this->mGlmDesign->GetDefaultWorkingDir();
}

/**
 * @return string
 */
string QdecProject::GetWorkingDir ( )
{
  return this->mGlmDesign->GetWorkingDir();
}


/**
 * @return 0 if ok, 1 on error
 * @param  isWorkingDir
 */
int QdecProject::SetWorkingDir ( const char* isWorkingDir )
{
  return this->mGlmDesign->SetWorkingDir( isWorkingDir );
}


/**
 * @return vector< string >
 */
vector< string > QdecProject::GetSubjectIDs ( )
{
  return this->mDataTable->GetSubjectIDs();
}


/**
 * @return vector< string >
 */
vector< string > QdecProject::GetDiscreteFactors ( )
{
  return this->mDataTable->GetDiscreteFactors();
}


/**
 * @return vector< string >
 */
vector< string > QdecProject::GetContinousFactors ( )
{
  return this->mDataTable->GetContinuousFactors();
}


/**
 * @return string
 */
string QdecProject::GetHemi ( )
{
  return this->mGlmDesign->GetHemi();
}


/**
 * From the given design parameters, this creates the input data required by
 * mri_glmfit:
 *  - the 'y' data (concatenated subject volumes)
 *  - the FSGD file
 *  - the contrast vectors, as .mat files
 * and writes this data to the specified working directory.
 * @return int
 * @param  isName
 * @param  isFirstDiscreteFactor
 * @param  isSecondDiscreteFactor
 * @param  isFirstContinuousFactor
 * @param  isSecondContinuousFactor
 * @param  isMeasure
 * @param  isHemi
 * @param  iSmoothnessLevel
 * @param  iProgressUpdateGUI
 */
int QdecProject::CreateGlmDesign ( const char* isName,
                                   const char* isFirstDiscreteFactor,
                                   const char* isSecondDiscreteFactor,
                                   const char* isFirstContinuousFactor,
                                   const char* isSecondContinuousFactor,
                                   const char* isMeasure,
                                   const char* isHemi,
                                   int iSmoothnessLevel,
                                   ProgressUpdateGUI* iProgressUpdateGUI )
{
  int errorCode;
  errorCode = this->mGlmDesign->Create ( this->mDataTable,
                                         isName,
                                         isFirstDiscreteFactor,
                                         isSecondDiscreteFactor,
                                         isFirstContinuousFactor,
                                         isSecondContinuousFactor,
                                         isMeasure,
                                         isHemi,
                                         iSmoothnessLevel,
                                         iProgressUpdateGUI );
  if( errorCode )
    {
    return errorCode;
    }

  if( iProgressUpdateGUI )
    {
    iProgressUpdateGUI->BeginActionWithProgress("Writing input files..." );
    }


  if( mGlmDesign->WriteFsgdFile() )
    {
    fprintf( stderr, "ERROR: QdecProject::CreateGlmDesign: "
             "could not create fsgd file\n");
    return(-3);
    }

  if( mGlmDesign->WriteContrastMatrices() )
    {
    fprintf( stderr, "ERROR: QdecProject::CreateGlmDesign: could not "
             "generate contrasts\n");
    return(-4);
    }

  if( mGlmDesign->WriteYdataFile() )
    {
    fprintf( stderr, "ERROR: QdecProject::CreateGlmDesign: could not "
             "create y.mgh file\n");
    return(-4);
    }

  if( iProgressUpdateGUI )
    {
    iProgressUpdateGUI->EndActionWithProgress();
    }

  return 0;
}

/**
 * @return int
 
int QdecProject::LoadGlmDesign(const char *fileName)
{
    int retval = this->mGlmDesign->ReadFsgdFile(fileName);
    if (retval != 0)
      {
      return retval;
      }
    
    // init the mGlmFitter
    retval = this->mGlmFitter->Load( this->mGlmDesign );
    
    return retval;
}
*/

/**
 * @return int
 */
int QdecProject::RunGlmFit ( )
{
  return this->mGlmFitter->Run( mGlmDesign );
}


/**
 * @return QdecGlmFitResults
 */
QdecGlmFitResults* QdecProject::GetGlmFitResults ( )
{
  return this->mGlmFitter->GetResults();
}


/**
 * Run mri_label2label on each subject, mapping the label that was drawn on 
 * the average surface onto each subject. Optionally supply a GUI manager
 * to allow posting progress info.
 * @return int
 * @param  ifnLabel
 * @param  iProgressUpdateGUI
 */
int QdecProject::GenerateMappedLabelForAllSubjects 
( const char* ifnLabel,
  ProgressUpdateGUI* iProgressUpdateGUI )
{
  vector< string > subjects = this->GetSubjectIDs();
  int numSubjects = this->GetSubjectIDs().size();
  float stepIncrement = 100.0 / numSubjects-1;
  int nStep = 1;

  if ( 0 == numSubjects )
    throw runtime_error( "Zero subjects! Cannot run mri_label2label\n" );

  if( iProgressUpdateGUI )
  {
    iProgressUpdateGUI->BeginActionWithProgress
      ( "Running mri_label2label..." );
  }
      
  for( int i=0; i < numSubjects; i++ )
  {
    // build a command line for this subject
    stringstream ssCommand;
    ssCommand << "mri_label2label"
              << " --srclabel " << ifnLabel
              << " --srcsubject " << this->GetAverageSubject()
              << " --trgsubject " << subjects[i]
              << " --trglabel " << ifnLabel
              << " --regmethod surface"
              << " --hemi " << this->GetHemi();

     // Now run the command.
    if( iProgressUpdateGUI )
    {
      string status = "Running mri_label2label on subject '";
      status += subjects[i];
      status += "'...";
      iProgressUpdateGUI->UpdateProgressMessage( status.c_str() );
      iProgressUpdateGUI->UpdateProgressPercent
        ( (float)nStep++ * stepIncrement );
    }
    char* sCommand = strdup( ssCommand.str().c_str() );
    printf( "\n----------------------------------------------------------\n" );
    printf( "%s\n", sCommand );
    fflush(stdout);fflush(stderr);
    int rRun = system( sCommand );
    if ( -1 == rRun )
      throw runtime_error( "system call failed: " + ssCommand.str() );
    if ( rRun > 0 )
      throw runtime_error( "command failed: " + ssCommand.str() );
    free( sCommand );
  }

  if( iProgressUpdateGUI )
  {
    iProgressUpdateGUI->UpdateProgressMessage( "Completed mri_label2label." );
    iProgressUpdateGUI->UpdateProgressPercent( 100 );
    iProgressUpdateGUI->EndActionWithProgress();
  }

  return 0;
}

/**
 * @return QdecGlmDesign
 */
QdecGlmDesign* QdecProject::GetGlmDesign ( ) 
{
  return this->mGlmDesign;
}

const char*
QdecProject::GetMetadataFileName () const {

  static char fnMetadata[] = "QdecProjectMetadata.txt";
  return fnMetadata;
}

