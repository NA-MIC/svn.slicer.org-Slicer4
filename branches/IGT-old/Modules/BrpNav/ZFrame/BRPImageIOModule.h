#ifndef __BRP_IMAGE_IO_MODULE_H__
#define __BRP_IMAGE_IO_MODULE_H__

#include <OpenTracker/OpenTracker.h>
#include <OpenTracker/dllinclude.h>
#include <OpenTracker/input/SPLModules.h>
#include <OpenTracker/input/BRPImageIOSink.h>
#include <OpenTracker/input/BRPImageIOSource.h>

#include <OpenTracker/types/Image.h>

#include <string>

namespace ot {

class OPENTRACKER_API BRPImageIOModule : public Module, public NodeFactory
{
 public:

 public:
  
  // Constructor and destructor
  BRPImageIOModule();
  virtual ~BRPImageIOModule();
  
  Node* createNode(const std::string& name,  ot::StringTable& attributes);
  void  pushEvent();
  void  pullEvent() {};
  void  init(StringTable&, ConfigNode *);

  void  setTracker(std::vector<float> pos,std::vector<float> quat);
  void  setImage(Image& img, int w, int h, std::vector<float> pos,std::vector<float> quat);

  inline BRPImageIOSink* getSink() { return sink; };

 private:
  
  BRPImageIOSink*   sink;
  BRPImageIOSource* source;
  
  friend class  BRPImageIOSink;
  friend class  BRPImageIOSource;
};

OT_MODULE(BRPImageIOModule);


} // end of namespace ot

#endif // __BRP_IMAGE_IO_MODULE_H__
