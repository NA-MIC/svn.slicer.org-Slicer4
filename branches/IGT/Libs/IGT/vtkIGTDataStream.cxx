
#include "vtkIGTDataStream.h"
#include "vtkObjectFactory.h"

// Constructors/Destructors
//  

vtkIGTDataStream* vtkIGTDataStream::New()
{
  vtkObject* ret=vtkObjectFactory::CreateInstance("vtkIGTDataStream");
  if(ret)
    {
      return(vtkIGTDataStream*) ret;
    }
  return new vtkIGTDataStream;
}

vtkIGTDataStream::vtkIGTDataStream ( ) {
initAttributes();
}

vtkIGTDataStream::~vtkIGTDataStream ( ) { }

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
 * Set the value of m_buffer_size
 * @param new_var the new value of m_buffer_size
 */
void vtkIGTDataStream::setBuffer_size ( int new_var ) {
  m_buffer_size = new_var;
}

/**
 * Get the value of m_buffer_size
 * @return the value of m_buffer_size
 */
int vtkIGTDataStream::getBuffer_size ( ) {
  return m_buffer_size;
}

/**
 * Set the value of m_LastInputNum
 * @param new_var the new value of m_LastInputNum
 */
void vtkIGTDataStream::setLastInputNum ( int new_var ) {
  m_LastInputNum = new_var;
}

/**
 * Get the value of m_LastInputNum
 * @return the value of m_LastInputNum
 */
int vtkIGTDataStream::getLastInputNum ( ) {
  return m_LastInputNum;
}

/**
 * Set the value of m_LastInputTime
 * @param new_var the new value of m_LastInputTime
 */
void vtkIGTDataStream::setLastInputTime ( int new_var ) {
  m_LastInputTime = new_var;
}

/**
 * Get the value of m_LastInputTime
 * @return the value of m_LastInputTime
 */
int vtkIGTDataStream::getLastInputTime ( ) {
  return m_LastInputTime;
}

// Other methods
//  





/**
 */
void vtkIGTDataStream::create_MRML_node ( ) {

}


/**
 */
void vtkIGTDataStream::update_mrml ( ) {

}

void vtkIGTDataStream::initAttributes ( ) {
}

