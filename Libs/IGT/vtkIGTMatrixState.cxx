
#include "vtkIGTMatrixState.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"

// Constructors/Destructors
//  


vtkIGTMatrixState* vtkIGTMatrixState::New()
{
  vtkObject* ret=vtkObjectFactory::CreateInstance("vtkIGTMatrixState");
  if(ret)
    {
      return(vtkIGTMatrixState*) ret;
    }
  return new vtkIGTMatrixState;
}

vtkIGTMatrixState::vtkIGTMatrixState ( ) {
  initAttributes();
}

vtkIGTMatrixState::~vtkIGTMatrixState ( ) { }

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
 * Set the value of m_timestamp
 * @param new_var the new value of m_timestamp
 */
void vtkIGTMatrixState::setTimestamp ( int new_var ) {
  m_timestamp = new_var;
}

/**
 * Get the value of m_timestamp
 * @return the value of m_timestamp
 */
int vtkIGTMatrixState::getTimestamp ( ) {
  return m_timestamp;
}

// Other methods
//  


/**
 * obtain registration result from IGTRegistration and copy it to
 * userspace_to_IJK_matrix
 */
void vtkIGTMatrixState::get_userspace_to_IJK_from_registration ( ) {

}


/**
 */
void vtkIGTMatrixState::get_userspace_to_RAS_from_registration ( ) {

}


/**
 */
void vtkIGTMatrixState::get_RAS_to_IJK_fom_Slicer ( ) {

}

void vtkIGTMatrixState::initAttributes ( ) {
}

