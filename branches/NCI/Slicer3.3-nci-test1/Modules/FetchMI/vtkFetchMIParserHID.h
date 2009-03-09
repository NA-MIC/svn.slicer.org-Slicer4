#ifndef __vtkFetchMIParserHID_h
#define __vtkFetchMIParserHID_h

#include "vtkObject.h"
#include "vtkFetchMIWin32Header.h"
#include "vtkXMLDataParser.h"
#include "vtkXMLDataElement.h"
#include "vtkFetchMIParser.h"
#include <string>


class VTK_FETCHMI_EXPORT vtkFetchMIParserHID : public vtkFetchMIParser
{
  
public:
  static vtkFetchMIParserHID* New();
  vtkTypeRevisionMacro(vtkFetchMIParserHID, vtkFetchMIParser);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Method returns the class's vtkXMLDataParser
  vtkGetObjectMacro ( Parser, vtkXMLDataParser );

  // Description:
  // Function for parsing the response to query for server-supported metadata
  virtual int ParseMetadataQueryResponse(const char *filename ) { return 0; };

  // Description:
  // Function for parsing the response to query for all values for any metadata attribute.
  virtual int ParseValuesforAttributeQueryResponse(const char *filename, const char *att ) { return 0; };

  // Description:
  // Function for parsing the response to query for resources on the server.
  virtual int ParseResourceQueryResponse(const char *filename ) { return 0; };

  // Description:
  // Function for parsing the response to requests that metadata be deleted on server
  virtual int ParseMetatdataDeleteResponse(const char *filename ) { return 0; };

  // Description:
  // Function for parsing the response to requests that a resource be deleted on server
  virtual int ParseResourceDeleteResponse(const char *filename ) { return 0; };

  // Description:
  // Function for parsing the response to posting of new metadata to server.
  virtual int ParseMetadataPostResponse( const char *filename) { return 0; };

  // Description:
  // Function for parsing the response to posting of reource to server.
  virtual int ParseResourcePostResponse(const char *filename ) { return 0; };

  // Description:
  // Virtual functions for parsing download requests.
  virtual int ParseResourceDownloadResponse(const char *filename ) { return 0; };
  
  // Description:
  // Virtual function for looking for errors in response to posts or queries on the server.
  virtual int ParseForErrors (const char *filename ) { return 0; };
  
  // Description:
  // Gets an XMLEntry and depending on the ResponseType the
  // parser is expecting, does the appropriate thing to process it.
  virtual void GetXMLEntry ( vtkXMLDataElement *element );
  
 protected:
  vtkFetchMIParserHID();
  virtual ~vtkFetchMIParserHID();

  vtkXMLDataParser *Parser;
  //BTX
  std::string ErrorString;
  //ETX

  vtkFetchMIParserHID(const vtkFetchMIParserHID&); // Not implemented
  void operator=(const vtkFetchMIParserHID&); // Not Implemented
};

#endif

