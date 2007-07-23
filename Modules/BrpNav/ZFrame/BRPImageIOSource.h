#ifndef __BRP_IMAGE_IO_SOURCE_H__
#define __BRP_IMAGE_IO_SOURCE_H__

#include <OpenTracker/OpenTracker.h>
#include <string>

namespace ot {

  class BRPImageIOSource : public Node
  {
  private:
    
  public:
    
    std::string StationName;
    
  public:
    BRPImageIOSource(std::string stationName) : changed(0) {};
    virtual int isEventGenerator() {return 1;};
    int changed;
    
    Event event;
  };
    
} // end of namespace ot

#endif // __BRP_IMAGE_IO_SOURCE_H__
