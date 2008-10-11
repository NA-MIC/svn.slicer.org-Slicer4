#ifdef WIN32
#pragma warning(disable:4786)
#pragma warning(disable:4290)
#endif

#include <OpenTracker/input/BRPImageIOModule.h>
#include <OpenTracker/core/Context.h>
#include <iostream>


namespace ot {

//------------------------------------------------------------------------------
// In following macro, module registration function named
// "registerModuleBRPImageIOModule()"  is defined.
// This would be called from SPLModule class.
//------------------------------------------------------------------------------
OT_MODULE_REGISTER_FUNC(BRPImageIOModule){
  OT_MODULE_REGISTRATION_DEFAULT(BRPImageIOModule, "BRPImageIOConfig" );
}


//------------------------------------------------------------------------------
// FUNCTION: BRPImageIOModlue()
//
// Constructor of this class
//------------------------------------------------------------------------------
BRPImageIOModule::BRPImageIOModule() 
{
  source    = NULL;
  sink      = NULL;
}


//------------------------------------------------------------------------------
// FUNCTION: BRPImageIOModlue()
//
// Destructor of this class
//------------------------------------------------------------------------------
BRPImageIOModule::~BRPImageIOModule()
{

}


//------------------------------------------------------------------------------
// FUNCTION: createNode()
//
// This function creates node for "Sink" and "Source".
// Called when NaviTrack finds a name of node in XML file during starting up.
// Note that this implementation does not allow to have multiple nodes.
// Use std::vector to manage multiple nodes.
//------------------------------------------------------------------------------
ot::Node * BRPImageIOModule::createNode( const std::string& name,  ot::StringTable& attributes)
{
  if( name.compare("BRPImageIOSink") == 0 )
    {
      std::string strName=attributes.get("name");
      std::cout << "BRPImageIOModule::createNode(): creating a BRPImageIOSink node" << std::endl;
      std::cout << "BRPImageIOModule::createNode(): attribute\"name\" is " << strName << std::endl;
      sink =  new BRPImageIOSink();
      return sink;
    }
  if(name.compare("BRPImageIOSource") == 0 )
    {
      std::string strName=attributes.get("name");
      std::cout << "BRPImageIOModule::createNode(): creating a BRPImageIOSource node" << std::endl;
      std::cout << "BRPImageIOModule::createNode(): attribute\"name\" is " << strName << std::endl;
      source = new BRPImageIOSource(strName);
      return source;
    }
  return NULL;
}


//------------------------------------------------------------------------------
// FUNCTION: pushState
//
// This function updates trackerinfo for all sources of this module
//------------------------------------------------------------------------------
void BRPImageIOModule::pushEvent()
{
  static short* pix = NULL;

  if(source)
    {
    }
}
  
  
//------------------------------------------------------------------------------
// FUNCTION: init()
//
// This function initializes the module.
//------------------------------------------------------------------------------
void BRPImageIOModule::init(StringTable& attributes, ConfigNode * localTree)
{
  std::cout << "BRPImageIOModule::init() is called." << std::endl;

  std::string strName=attributes.get("name");
  std::cout << "BRPImageIOModule::init(): attribute \"name\" is " 
            << strName << std::endl;
}


//------------------------------------------------------------------------------
// FUNCTION: setTracker()
//
// This function updates trackerinfo for all sources of this module
//------------------------------------------------------------------------------
void BRPImageIOModule::setTracker(std::vector<float> pos,std::vector<float> quat)
{
  std::cout << "BRPImageIOModule::setTracker() is called." << std::endl;

  if (pos.size() != 3 || quat.size() != 4) {
    std::cout << "BRPImageIOModule::setTracker(): illegal vector size." << std::endl;
    return;
  }

  if (source!=NULL)
    {
      ot::Event *event = new ot::Event();
      event->setAttribute("command", std::string("ZFRAME"));
      event->setAttribute("position",pos);
      event->setAttribute("orientation",quat);
      event->getButton()=0;
      event->getConfidence()=1.0;
      source->changed=1;
      event->timeStamp();
      source->updateObservers( *event );
    }
}


//------------------------------------------------------------------------------
// FUNCTION: setImage()
//
// This function updates image for all sources of this module
//------------------------------------------------------------------------------
void BRPImageIOModule::setImage(Image& img, int w, int h,
                                std::vector<float> pos,std::vector<float> quat)
{
  //static short* pix = NULL;
  
  if(source!=NULL)
    {
      //ot::Event event;
      ot::Event* event = new ot::Event();
      event->setAttribute("command", std::string("ZFRAME"));
      event->setAttribute("xsize", w);
      event->setAttribute("ysize", h);
      event->setAttribute("image", img);
      event->setAttribute("position",pos);
      event->setAttribute("orientation",quat);
      std::cout << "MyTutorialModule::setImage() is called. (x, y) = (" 
                << w << ", " << h << ")" << std::endl;
      //source->changed=1;
      event->timeStamp();
      source->updateObservers( *event );
    }

  /*
  if(source!=NULL)
    {
      // split a image
      int nlines  = MAXIMUM_IMAGE_SIZE / (w*sizeof(short));
      int nblocks = h / nlines;
      if (h % nlines > 0) nblocks ++;

      for (int n = 0; n < nblocks; n ++) 
        {
          int ysize;
          int endflag = 0;

          if (n == nblocks-1)
            {
              ysize = h-nlines*n;
              endflag = 1;
            }
          else
            ysize = nlines;

          short* ba = (short*)img.image_ptr;
          Image fragment(w,ysize,sizeof(short),&ba[w*n*nlines]);
          
          //ot::Event event;
          ot::Event event;
          event.setAttribute("endflag",   endflag);
          event.setAttribute("xsize",     w);
          event.setAttribute("ysize",     ysize);
          event.setAttribute("totalysize",h);
          event.setAttribute("line",      n*nlines);
          event.setAttribute("image",     fragment);
          event.setAttribute("index",     n);

          event.timeStamp();
          source->updateObservers( event );
          this->context->loopOnce();
          //usleep(pushImageInterval);
          OSUtils::sleep(pushImageInterval);
        }
    }
  */
}


} // end of "namespace ot"





