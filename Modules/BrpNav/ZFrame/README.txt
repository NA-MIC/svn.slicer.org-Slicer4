--------------------------------------------------------------------------------

 Z-frame Tracking Program (based on Simon DiMaio's ZTrackerTransform)

--------------------------------------------------------------------------------


0. Requrirements
=================

 Following libraries must be installed before building Z-frame Tracking Program.

  - NaviTrack
  - Insight Segmentation and Registration Toolkit (http://www.itk.org/)


1. INSTALL
===========

(1)Add BRPImageIOModule to the existing NaviTrack

  - Copy module files:

     $ cd <ZFrame dir>
     $ cp BRPImageIO*.cxx <NaviTrack dir>/src/input/
     $ cp BRPImageIO*.h   <NaviTrack dir>/include/OpenTracker/input/

  - Edit <NaviTrack dir>/CMakeList.txt:
   Add following lines in SET(BASESRCS ...) section:

     src/input/BRPImageIOModule.cxx
     src/input/BRPImageIOSink.cxx
    
  - Edit <NaviTrack dir>/src/misc/SPLModules.cxx:
   Add following include statements:

     #include <OpenTracker/input/BRPImageIOModule.h>

   and following line into addSPLModules() function:

     OT_REGISTER_MODULE(BRPImageIOModule, NULL);

  - Run CMake and rebuild NaviTrack

     $ ccmake .
     <configuration screen appears. press c, and g to generate
      a new Makefile.>
     $ make


(2)Edit following lines in Makefile:

   NTDIR  = <NaviTrack directory>
   ITKDIR = <Insight Toolkit directory>

(3)Bulid:

    $ cd <ZFrame dir>
    $ make


3. Run the program
===================

(1)Acquire Z-frame image:

 Currently, 256x256 matrix and 160mm FOV are assumed.


(2)Configure NaviTrack xml file:

 Example code is in send_zframe.xml


(3)Set LD_LIBRARY_PATH environmental variable to use libNaviTrack.so

  in csh:
    $setenv LD_LIBRARY_PATH <NaviTrack dir>

  in bash:
    $export LD_LIBRARY_PATH <NaviTrack dir>


(4)Run from a console as:

    $ ./zframe <XML file> <DICOM file>

