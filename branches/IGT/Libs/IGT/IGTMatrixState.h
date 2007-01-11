
#ifndef IGTAURORATRACKER_H
#define IGTAURORATRACKER_H

#include <string>

/**
  * class IGTAuroraTracker
  */

class IGTAuroraTracker
{
public:

  // Constructors/Destructors
  //  


  /**
   * Empty Constructor
   */
  IGTAuroraTracker ( );

  /**
   * Empty Destructor
   */
  virtual ~IGTAuroraTracker ( );

  // Static Public attributes
  //  

  // Public attributes
  //  


  // Public attribute accessor methods
  //  


  // Public attribute accessor methods
  //  


protected:

  // Static Protected attributes
  //  

  // Protected attributes
  //  


  // Protected attribute accessor methods
  //  


  // Protected attribute accessor methods
  //  


private:

  // Static Private attributes
  //  

  // Private attributes
  //  


  // Private attribute accessor methods
  //  


  // Private attribute accessor methods
  //  



};

#endif // IGTAURORATRACKER_H


#ifndef IGTMATRIXSTATE_H
#define IGTMATRIXSTATE_H

#include <string>
#include <vector>



/**
  * class IGTMatrixState
  * This represents a single data from sensor source
  */

class IGTMatrixState
{
public:

  // Constructors/Destructors
  //  


  /**
   * Empty Constructor
   */
  IGTMatrixState ( );

  /**
   * Empty Destructor
   */
  virtual ~IGTMatrixState ( );

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


private:

  // Static Private attributes
  //  

  // Private attributes
  //  

  vtkMatrix4X4 m_location_matrix_in_userspace;
  vtkMatrix4X4 m_matrix_in_RAS;
  vtkMatrix4X4 m_matrix_in_IJK;
  void m_timestamp;

  // Private attribute accessor methods
  //  


  // Private attribute accessor methods
  //  


  /**
   * Set the value of m_location_matrix_in_userspace
   * @param new_var the new value of m_location_matrix_in_userspace
   */
  void setLocation_matrix_in_userspace ( vtkMatrix4X4 new_var );

  /**
   * Get the value of m_location_matrix_in_userspace
   * @return the value of m_location_matrix_in_userspace
   */
  vtkMatrix4X4 getLocation_matrix_in_userspace ( );


  /**
   * Set the value of m_matrix_in_RAS
   * @param new_var the new value of m_matrix_in_RAS
   */
  void setMatrix_in_RAS ( vtkMatrix4X4 new_var );

  /**
   * Get the value of m_matrix_in_RAS
   * @return the value of m_matrix_in_RAS
   */
  vtkMatrix4X4 getMatrix_in_RAS ( );


  /**
   * Set the value of m_matrix_in_IJK
   * @param new_var the new value of m_matrix_in_IJK
   */
  void setMatrix_in_IJK ( vtkMatrix4X4 new_var );

  /**
   * Get the value of m_matrix_in_IJK
   * @return the value of m_matrix_in_IJK
   */
  vtkMatrix4X4 getMatrix_in_IJK ( );


  /**
   * Set the value of m_timestamp
   * @param new_var the new value of m_timestamp
   */
  void setTimestamp ( void new_var );

  /**
   * Get the value of m_timestamp
   * @return the value of m_timestamp
   */
  void getTimestamp ( );



  /**
   */
  void get_RAS_to_IJK_fom_Slicer ( );

  void initAttributes ( ) ;

};

#endif // IGTMATRIXSTATE_H
