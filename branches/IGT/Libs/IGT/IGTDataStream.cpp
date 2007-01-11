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



#include "igtdatastream.h"

// Constructors/Destructors
//  

IGTDataStream::IGTDataStream ( ) {
initAttributes();
}

IGTDataStream::~IGTDataStream ( ) { }

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
void IGTDataStream::setBuffer_size ( int new_var ) {
  m_buffer_size = new_var;
}

/**
 * Get the value of m_buffer_size
 * @return the value of m_buffer_size
 */
int IGTDataStream::getBuffer_size ( ) {
  return m_buffer_size;
}

/**
 * Set the value of m_LastInputNum
 * @param new_var the new value of m_LastInputNum
 */
void IGTDataStream::setLastInputNum ( int new_var ) {
  m_LastInputNum = new_var;
}

/**
 * Get the value of m_LastInputNum
 * @return the value of m_LastInputNum
 */
int IGTDataStream::getLastInputNum ( ) {
  return m_LastInputNum;
}

/**
 * Set the value of m_LastInputTime
 * @param new_var the new value of m_LastInputTime
 */
void IGTDataStream::setLastInputTime ( int new_var ) {
  m_LastInputTime = new_var;
}

/**
 * Get the value of m_LastInputTime
 * @return the value of m_LastInputTime
 */
int IGTDataStream::getLastInputTime ( ) {
  return m_LastInputTime;
}

// Other methods
//  


/**
 */
void IGTDataStream::register_matrix_state_and_create_buffer ( ) {

}


/**
 */
void IGTDataStream::create_MRML_node ( ) {

}


/**
 */
void IGTDataStream::update_mrml ( ) {

}

void IGTDataStream::initAttributes ( ) {
}

