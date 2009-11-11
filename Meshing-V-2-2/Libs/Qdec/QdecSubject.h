/**
 * @file  QdecSubject.h
 * @brief Stores all data associated with a subject.
 *
 * This is one row from the input data table file (qdec.table.dat).
 */
/*
 * Original Author: Nick Schmansky
 */

#ifndef QDECSUBJECT_H
#define QDECSUBJECT_H



#include <QdecConfigure.h>
#include "vtkQdecWin32Header.h"

#include <string>
#include <vector>

#include "QdecFactor.h"
//#include "vtkQdec.h"
//BTX
using namespace std;

class VTK_Qdec_EXPORT QdecSubject
{
public:

  // Constructors/Destructors
  //

  QdecSubject ( string isId, vector < QdecFactor* > iFactors );

  virtual ~QdecSubject ( );

  /**
   * Get the value of msId
   * the subject identifier, as found in the 'fsid' column of the
   * table.dat input file.
   * @return the value of msId
   * @return string
   */
  string GetId ( );


  /**
   * @return string
   * @param  isFactorName
   */
  string GetDiscreteFactor ( const char* isFactorName );


  /**
   * @return double
   * @param  isFactorName
   */
  double GetContinuousFactor ( const char* isFactorName );

  /**
   * @return vector < QdecFactor* >
   */
  vector < QdecFactor* > GetContinuousFactors ( );

  /**
   * @return vector < QdecFactor* >
   */
  vector < QdecFactor* > GetFactors ( );

private:

  // private attributes
  //

  // the subject identifier, as found in the 'fsid' column
  // of the table.dat input file.
  string msId;

  // Stores factor values (either discrete or continous)
  // pertaining to this subject.
  vector < QdecFactor* > mFactors;

};
//ETX
#endif // QDECSUBJECT_H

