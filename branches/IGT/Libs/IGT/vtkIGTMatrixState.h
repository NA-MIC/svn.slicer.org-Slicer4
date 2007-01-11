

#ifndef IGTMATRIXSTATE_H
#define IGTMATRIXSTATE_H

#include <string>
#include <vector>

#include "vtkObject.h"
#include "vtkMatrix4x4.h"

/**
  * class vtkIGTMatrixState
  * This represents a single data from sensor source
  */

class vtkIGTMatrixState : public vtkObject
{
public:

  // Constructors/Destructors
  //  
    static vtkIGTMatrixState *New();
      vtkTypeMacro(vtkIGTMatrixState,vtkObject);
      void PrintSelf(ostream& os, vtkIndent indent);

  /**
   * Empty Constructor
   */
  vtkIGTMatrixState ( );
  void Init();
  /**
   * Empty Destructor
   */
  virtual ~vtkIGTMatrixState ( );



protected:

  vtkMatrix4x4* Matrix;
  vtkMatrix4x4* Matrix_in_RAS;
  vtkMatrix4x4* Matrix_in_IJK;
  int m_timestamp;
  // Private attribute accessor methods
  //  




  /**
   * Set the value of m_timestamp
   * @param new_var the new value of m_timestamp
   */
  void setTimestamp ( int new_var );

  /**
   * Get the value of m_timestamp
   * @return the value of m_timestamp
   */
  int getTimestamp ( );



  

};

#endif // IGTMATRIXSTATE_H
