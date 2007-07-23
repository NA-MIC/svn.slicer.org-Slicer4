#ifndef __BRP_IMAGE_IO_SINK_H__
#define __BRP_IMAGE_IO_SINK_H__

#include <OpenTracker/OpenTracker.h>
#include <OpenTracker/types/Image.h>

namespace ot {
  
  class BRPImageIOSink : public ot::Node
  {
  private:
    std::vector<float> position;
    std::vector<float> orientation;

    int workphase;

    // This node has two bufferes for received data.
    Image image;
    int   width;
    int   height;

  public:
    
    BRPImageIOSink();
    ~BRPImageIOSink();
    
  public:
    
    virtual int isEventGenerator() {return 1;};
    virtual void onEventGenerated(Event&, Node&);

    inline void getTracker(std::vector<float>& pos, std::vector<float>& ori)
    {
      for (int i = 0; i < 3; i ++) pos[i] = position[i];
      for (int i = 0; i < 4; i ++) ori[i] = orientation[i];
    }

    inline void getImage(Image& im, int& w, int& h)
    {
      im = image;
      w  = width;
      h  = height;
    }

  };

} // end of namespace ot

#endif // end of __BRP_IMAGE_IO_SINK_H__
