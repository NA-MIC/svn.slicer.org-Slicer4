

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

  /**
   * Empty Destructor
   */
  virtual ~vtkIGTMatrixState ( );

  // Static Public attributes
  //  

  // Public attributes
  //  


  // Public attribute accessor methods
  //  


  // Public attribute accessor methods
  //  



  /**
   * obtain registration result from IGTRegistration and copy it to
   * userspace_to_IJK_matrix
   */
  void get_userspace_to_IJK_from_registration ( );


  /**
   */
  void get_userspace_to_RAS_from_registration ( );

protected:

  // Static Protected attributes
  //  

  // Protected attributes
  //  


  // Protected attribute accessor methods
  //  


  // Protected attribute accessor methods
  //  


  // Static Private attributes
  //  

  // Private attributes
  //  

  vtkMatrix4x4* m_location_matrix_in_userspace;
  vtkMatrix4x4* m_matrix_in_RAS;
  vtkMatrix4x4* m_matrix_in_IJK;
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



  /**
   */
  void get_RAS_to_IJK_fom_Slicer ( );

  void initAttributes ( ) ;

};

#endif // IGTMATRIXSTATE_H
