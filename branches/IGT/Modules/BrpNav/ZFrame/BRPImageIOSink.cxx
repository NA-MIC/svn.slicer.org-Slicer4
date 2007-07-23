#include <OpenTracker/input/BRPImageIOSink.h>
#include <OpenTracker/input/BRPImageIOModule.h>

#ifdef WIN32
#include <windows.h>
#include <OpenTracker/misc/SPLOT_Win.h>
#endif

#include <iostream>



namespace ot {

  BRPImageIOSink::BRPImageIOSink()
  {
    position.resize(3);
    orientation.resize(4);

    position[0] = position[1] = position[2] = 0.0;
    orientation[0] = orientation[1] = orientation[2] = orientation[3] = 0.0;

    width = height = 0;
  }

  BRPImageIOSink::~BRPImageIOSink()
  {
  }

  void BRPImageIOSink::onEventGenerated( Event& event, Node& generator)
  {

    std::cerr << "BRPImageIOSink::onEventGenerated()" << std::endl;

    if (event.hasAttribute("image") &&
        event.hasAttribute("xsize") &&
        event.hasAttribute("ysize"))
      {
        //image.SetSize(width, height, sizeof(short));
        image = event.getAttribute((Image*)NULL, "image");
        width = event.getAttribute(std::string("xsize"),0);
        height = event.getAttribute(std::string("ysize"),0);
      }
    else
      {
        width  = 0;
        height = 0;
      }

    if (event.hasAttribute("position"))
      {
        for(int i = 0; i < 3; i ++)
          position[i]=event.getPosition()[i];
      }
    else
      {
        position[0]=0.0;
        position[1]=0.0;
        position[2]=0.0;
      }

    if (event.hasAttribute("orientation"))
      {
        for  (int i = 0; i < 4; i ++) {
          orientation[i]= event.getOrientation()[i];
        }
      }
    else
      {
        orientation[0]=0.0;
        orientation[1]=0.0;
        orientation[2]=0.0;
        orientation[3]=0.0;
      }
    std::cerr << "BRPImageIOSink = " << position[0]
              << ", " << position[1]
              << ", " << position[2] << std::endl;
    
  }


} // end of namespace ot
