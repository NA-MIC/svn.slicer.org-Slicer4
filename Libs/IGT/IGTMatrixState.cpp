#include "igtauroratracker.h"

// Constructors/Destructors
//  

IGTAuroraTracker::IGTAuroraTracker ( ) {
}

IGTAuroraTracker::~IGTAuroraTracker ( ) { }

//  
// Methods
//  


// Accessor methods
//  


// Other methods
//  



#include "igtmatrixstate.h"

// Constructors/Destructors
//  

IGTMatrixState::IGTMatrixState ( ) {
initAttributes();
}

IGTMatrixState::~IGTMatrixState ( ) { }

//  
// Methods
//  


// Accessor methods
//  


// Public static attribute accessor methods
//  


// Public attribute accessor methods
//  


// Protected static attribute accessor methods
//  


// Protected attribute accessor methods
//  


// Private static attribute accessor methods
//  


// Private attribute accessor methods
//  


/**
 * Set the value of m_location_matrix_in_userspace
 * @param new_var the new value of m_location_matrix_in_userspace
 */
void IGTMatrixState::setLocation_matrix_in_userspace ( vtkMatrix4X4 new_var ) {
  m_location_matrix_in_userspace = new_var;
}

/**
 * Get the value of m_location_matrix_in_userspace
 * @return the value of m_location_matrix_in_userspace
 */
vtkMatrix4X4 IGTMatrixState::getLocation_matrix_in_userspace ( ) {
  return m_location_matrix_in_userspace;
}

/**
 * Set the value of m_matrix_in_RAS
 * @param new_var the new value of m_matrix_in_RAS
 */
void IGTMatrixState::setMatrix_in_RAS ( vtkMatrix4X4 new_var ) {
  m_matrix_in_RAS = new_var;
}

/**
 * Get the value of m_matrix_in_RAS
 * @return the value of m_matrix_in_RAS
 */
vtkMatrix4X4 IGTMatrixState::getMatrix_in_RAS ( ) {
  return m_matrix_in_RAS;
}

/**
 * Set the value of m_matrix_in_IJK
 * @param new_var the new value of m_matrix_in_IJK
 */
void IGTMatrixState::setMatrix_in_IJK ( vtkMatrix4X4 new_var ) {
  m_matrix_in_IJK = new_var;
}

/**
 * Get the value of m_matrix_in_IJK
 * @return the value of m_matrix_in_IJK
 */
vtkMatrix4X4 IGTMatrixState::getMatrix_in_IJK ( ) {
  return m_matrix_in_IJK;
}

/**
 * Set the value of m_timestamp
 * @param new_var the new value of m_timestamp
 */
void IGTMatrixState::setTimestamp ( void new_var ) {
  m_timestamp = new_var;
}

/**
 * Get the value of m_timestamp
 * @return the value of m_timestamp
 */
void IGTMatrixState::getTimestamp ( ) {
  return m_timestamp;
}

// Other methods
//  


/**
 * obtain registration result from IGTRegistration and copy it to
 * userspace_to_IJK_matrix
 */
void IGTMatrixState::get_userspace_to_IJK_from_registration ( ) {

}


/**
 */
void IGTMatrixState::get_userspace_to_RAS_from_registration ( ) {

}


/**
 */
void IGTMatrixState::get_RAS_to_IJK_fom_Slicer ( ) {

}

void IGTMatrixState::initAttributes ( ) {
}

