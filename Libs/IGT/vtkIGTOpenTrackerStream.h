

#ifndef IGTOPENTRACKERSTREAM_H
#define IGTOPENTRACKERSTREAM_H

#include <string>
#include "vtkIGTDataStream.h"

/**
  * class IGTOpenTrackerStream
  */

class vtkIGTOpenTrackerStream : public vtkIGTDataStream
{
public:

  // Constructors/Destructors
  //  

    static vtkIGTOpenTrackerStream *New();
      vtkTypeMacro(vtkIGTOpenTrackerStream,vtkIGTDataStream);
      void PrintSelf(ostream& os, vtkIndent indent);

  /**
   * Empty Constructor
   */
  vtkIGTOpenTrackerStream ( );

  /**
   * Empty Destructor
   */
  virtual ~vtkIGTOpenTrackerStream ( );

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
  void initi_open_tracker ( );


  /**
   */
  void add_data_to_data_stream ( );


  /**
   */
  void callback_function ( );


  /**
   */
  void read_configuration_file ( );

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

#endif // IGTOPENTRACKERSTREAM_H
