
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


#ifndef IGTDATASTREAM_H
#define IGTDATASTREAM_H

#include <string>
#include <vector>



/**
  * class IGTDataStream
  */

class IGTDataStream
{
public:

  // Constructors/Destructors
  //  


  /**
   * Empty Constructor
   */
  IGTDataStream ( );

  /**
   * Empty Destructor
   */
  virtual ~IGTDataStream ( );

  // Static Public attributes
  //  

  // Public attributes
  //  


  // Public attribute accessor methods
  //  


  // Public attribute accessor methods
  //  



  /**
   */
  void register_matrix_state_and_create_buffer ( );


  /**
   */
  void create_MRML_node ( );


  /**
   */
  void update_mrml ( );

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

  int m_buffer_size;
  int m_LastInputNum;
  int m_LastInputTime;

  // Private attribute accessor methods
  //  


  // Private attribute accessor methods
  //  


  /**
   * Set the value of m_buffer_size
   * @param new_var the new value of m_buffer_size
   */
  void setBuffer_size ( int new_var );

  /**
   * Get the value of m_buffer_size
   * @return the value of m_buffer_size
   */
  int getBuffer_size ( );


  /**
   * Set the value of m_LastInputNum
   * @param new_var the new value of m_LastInputNum
   */
  void setLastInputNum ( int new_var );

  /**
   * Get the value of m_LastInputNum
   * @return the value of m_LastInputNum
   */
  int getLastInputNum ( );


  /**
   * Set the value of m_LastInputTime
   * @param new_var the new value of m_LastInputTime
   */
  void setLastInputTime ( int new_var );

  /**
   * Get the value of m_LastInputTime
   * @return the value of m_LastInputTime
   */
  int getLastInputTime ( );


  void initAttributes ( ) ;

};

#endif // IGTDATASTREAM_H
