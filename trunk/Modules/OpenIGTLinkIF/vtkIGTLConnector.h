/*==========================================================================

  Portions (c) Copyright 2008 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $HeadURL: $
  Date:      $Date: $
  Version:   $Revision: $

==========================================================================*/

#ifndef __vtkIGTLConnector_h
#define __vtkIGTLConnector_h

#include <string>
#include <map>
#include <vector>
#include <set>

#include "vtkObject.h"
#include "vtkOpenIGTLinkIFWin32Header.h" 
#include "igtlServerSocket.h"
#include "igtlClientSocket.h"


//class vtkSocketCommunicator;
class vtkMultiThreader;
class vtkMutexLock;

class vtkImageData;
class vtkMatrix4x4;

class vtkIGTLCircularBuffer;
class vtkMRMLNode;

class VTK_OPENIGTLINKIF_EXPORT vtkIGTLConnector : public vtkObject
{
 public:  
  //----------------------------------------------------------------
  // Constants Definitions
  //----------------------------------------------------------------

  //BTX
  enum {
    TYPE_NOT_DEFINED,
    TYPE_SERVER,
    TYPE_CLIENT,
    NUM_TYPE
  };
  
  enum {
    STATE_OFF,
    STATE_WAIT_CONNECTION,
    STATE_CONNECTED,
    NUM_STATE
  };

  enum {
    IO_UNSPECIFIED = 0x00,
    IO_INCOMING   = 0x01,
    IO_OUTGOING   = 0x02,
  };
  //ETX
  
  //BTX
  typedef struct {
    std::string   name;
    std::string   type;
    int           io;
    //vtkMRMLNode*  node;
  } DeviceInfoType;

  typedef std::map<int, DeviceInfoType> DeviceInfoListType;   // Device list:  index is referred as
                                                              // a device id in the connector.
  typedef std::set<int> DeviceIDSetType;

  typedef std::map<std::string, std::string> DeviceNameList;  // will be obsoleted
  typedef std::map<std::string, vtkMRMLNode*> MRMLNodeList;   // will be obsoleted
  //ETX

 public:
  
  static vtkIGTLConnector *New();
  vtkTypeRevisionMacro(vtkIGTLConnector,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  vtkGetMacro( ServerPort, int );
  vtkSetMacro( ServerPort, int );
  vtkGetMacro( Type, int );
  vtkSetMacro( Type, int );
  vtkGetMacro( State, int );
  //vtkSetMacro( State, int );

  vtkSetMacro( RestrictDeviceName, int );
  vtkGetMacro( RestrictDeviceName, int );

  //BTX
  void SetName (const char* str) { this->Name = str; }
  void SetName (std::string str) { this->Name = str; }
  const char* GetName() { return this->Name.c_str(); }
  void SetServerHostname(const char* str) { this->ServerHostname = str; }
  void SetServerHostname(std::string str) { this->ServerHostname = str; }
  const char* GetServerHostname() { return this->ServerHostname.c_str(); }
  //ETX

  //----------------------------------------------------------------
  // Constructor and Destructor
  //----------------------------------------------------------------

  vtkIGTLConnector();
  virtual ~vtkIGTLConnector();

  //----------------------------------------------------------------
  // Connector configuration
  //----------------------------------------------------------------

  int SetTypeServer(int port);
  int SetTypeClient(char* hostname, int port);
  //BTX
  int SetTypeClient(std::string hostname, int port);
  //ETX

  //----------------------------------------------------------------
  // Thread Control
  //----------------------------------------------------------------

  int Start();
  int Stop();
  static void* ThreadFunction(void* ptr);

  //----------------------------------------------------------------
  // OpenIGTLink Message handlers
  //----------------------------------------------------------------
  //BTX
  //igtl::ClientSocket::Pointer WaitForConnection();
  //ETX
  int WaitForConnection();
  int ReceiveController();
  int SendData(int size, unsigned char* data);
  int Skip(int length, int skipFully=1);

  //----------------------------------------------------------------
  // Circular Buffer
  //----------------------------------------------------------------

  //BTX
  typedef std::vector<std::string> NameListType;
  int GetUpdatedBuffersList(NameListType& nameList);
  vtkIGTLCircularBuffer* GetCircularBuffer(std::string& key);
  //ETX

  //----------------------------------------------------------------
  // Device Lists
  //----------------------------------------------------------------

  int GetDeviceID(const char* deviceName, const char* deviceType);
  int RegisterNewDevice(const char* deviceName, const char* deviceType, int io=IO_UNSPECIFIED);
  int UnregisterDevice(const char* deviceName, const char* deviceType, int io=IO_UNSPECIFIED);
  int UnregisterDevice(int id);
  int RegisterDeviceIO(int id, int io);

  //BTX
  DeviceInfoListType* GetDeviceInfoList()    { return &DeviceInfoList;         };
  DeviceIDSetType*    GetIncomingDevice()    { return &IncomingDeviceIDSet;    }
  DeviceIDSetType*    GetOutgoingDevice()    { return &OutgoingDeviceIDSet;    }
  DeviceIDSetType*    GetUnspecifiedDevice() { return &UnspecifiedDeviceIDSet; }
  //ETX

  /** old stuff */
  /*
  DeviceNameList* GetIncomingDeviceList() { return &IncomingDeviceList; };
  DeviceNameList* GetOutgoingDeviceList() { return &OutgoingDeviceList; };

  MRMLNodeList*   GetIncomingMRMLNodeList() { return &IncomingMRMLNodeList; };
  MRMLNodeList*   GetOutgoingMRMLNodeList() { return &OutgoingMRMLNodeList; };

  DeviceNameList* GetUnspecifiedDeviceList() { return &UnspecifiedDeviceList; };
  */


 private:
  //----------------------------------------------------------------
  // Connector configuration 
  //----------------------------------------------------------------
  //BTX
  std::string Name;
  //ETX
  int Type;
  int State;

  //----------------------------------------------------------------
  // Thread and Socket
  //----------------------------------------------------------------

  vtkMultiThreader* Thread;
  vtkMutexLock*     Mutex;
  //BTX
  igtl::ServerSocket::Pointer  ServerSocket;
  igtl::ClientSocket::Pointer  Socket;
  //ETX
  int               ThreadID;
  int               ServerPort;
  int               ServerStopFlag;

  //BTX
  std::string       ServerHostname;
  //ETX

  //----------------------------------------------------------------
  // Data
  //----------------------------------------------------------------

  //BTX
  typedef std::map<std::string, vtkIGTLCircularBuffer*> CircularBufferMap;
  CircularBufferMap Buffer;
  //ETX

  vtkMutexLock* CircularBufferMutex;
  
  int     RestrictDeviceName;  // Flag to restrict incoming and outgoing data by device names
  //BTX
  // -- Device Name (same as MRML node) and data type (data type string defined in OpenIGTLink)

  int                LastID;
  DeviceInfoListType DeviceInfoList;

  DeviceIDSetType   IncomingDeviceIDSet;
  DeviceIDSetType   OutgoingDeviceIDSet;
  DeviceIDSetType   UnspecifiedDeviceIDSet;

  /** old stuff **/
  /*
  DeviceNameList IncomingDeviceList;
  DeviceNameList OutgoingDeviceList;
  DeviceNameList UnspecifiedDeviceList;

  MRMLNodeList   IncomingMRMLNodeList;
  MRMLNodeList   OutgoingMRMLNodeList;
  */

  //ETX

};

#endif // __vtkIGTLConnector_h
