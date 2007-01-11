
#ifndef IGTDATASTREAM_H
#define IGTDATASTREAM_H


#include "vtkObject.h"


class vtkIGTDataStream : public vtkObject
{
public:

  // Constructors/Destructors
  //  
    static vtkIGTDataStream *New();
      vtkTypeMacro(vtkIGTDataStream,vtkObject);
      void PrintSelf(ostream& os, vtkIndent indent);


  /**
   * Empty Constructor
   */
  vtkIGTDataStream ( );

  /**
   * Empty Destructor
   */
  virtual ~vtkIGTDataStream ( );

  // Static Public attributes
  //  

  // Public attributes
  //  


  // Public attribute accessor methods
  //  


  // Public attribute accessor methods
  //  


protected:

  /**
   */
  void create_MRML_node ( );


  /**
   */
  void update_mrml ( );



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
